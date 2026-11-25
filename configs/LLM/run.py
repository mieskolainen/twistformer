# Scientific text completion with causal backbones and XGBoost specialists
#
# mikael.mieskolainen@cern.ch (2026)

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models as token_models, pre_tokenizers, trainers
import xgboost as xgb
import yaml

from models import BOS, EOS, PAD, adjust, backbone, expert_margins

DELIMITERS = {")": "(", "]": "[", "}": "{"}
COMPLETIONS = {
    "equation": re.compile(r"(?<!\\)\$(?!\$)([^$\n]{1,240}\$)"),
    "citation": re.compile(r"\\cite\{([^{}\n]{1,120}\})"),
}


def read_documents(files):
    """Read disjoint JSONL documents and reject duplicate IDs and text"""
    documents = {}
    ids, hashes = set(), set()
    for split, filename in files.items():
        documents[split] = []
        with Path(filename).open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                key, body = row["document_id"], row["text"]
                if not isinstance(key, str) or not key or not isinstance(body, str) or not body.strip():
                    raise ValueError(f"Expected a document_id and nonempty text in {filename}")
                digest = hashlib.sha256(" ".join(body.split()).encode()).digest()
                if key in ids or digest in hashes:
                    raise ValueError(f"Duplicate document in {filename}: {key}")
                ids.add(key)
                hashes.add(digest)
                documents[split].append(body)
        if not documents[split]:
            raise ValueError(f"Empty document split: {filename}")
    return documents


def tokenize(documents, options):
    """Fit one byte-level tokenizer and encode each document independently"""
    tokenizer = Tokenizer(token_models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=options["vocabulary_size"], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<pad>", "<bos>", "<eos>"])
    tokenizer.train_from_iterator(documents[options["fit_split"]], trainer=trainer)
    tokens = {
        split: [np.asarray([BOS, *item.ids, EOS], dtype=np.int32) for item in tokenizer.encode_batch(texts)]
        for split, texts in documents.items()
    }
    return tokenizer, tokens


def windows(documents, context):
    """Split the documents so each next-token target is counted once"""
    return [doc[i:i+context+1] for doc in documents for i in range(0, len(doc)-1, context)]


def batch(windows, device):
    """Pad right and reserve -100 for targets outside the scored sequence"""
    width = max(len(row)-1 for row in windows)
    inputs = torch.full((len(windows), width), PAD, dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for i, row in enumerate(windows):
        inputs[i, :len(row)-1] = torch.from_numpy(row[:-1])
        targets[i, :len(row)-1] = torch.from_numpy(row[1:])
    return inputs.to(device), targets.to(device)


def train(model, data, config, device, seed):
    """Train to an exact budget of next-token targets"""
    settings = config["training"]
    target = settings["training_tokens_per_backbone"]
    size = settings["batch_size"]
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    seen, steps = 0, 0
    start = perf_counter()
    model.train()
    while seen < target:
        order = rng.permutation(len(data))
        for first in range(0, len(data), size):
            items = [data[i] for i in order[first:first+size]]
            inputs, labels = batch(items, device)
            valid = sum(len(item)-1 for item in items)
            remaining = target-seen
            if valid > remaining:
                consumed = labels.ne(-100).reshape(-1).cumsum(0).reshape_as(labels)
                labels.masked_fill_(consumed > remaining, -100)
            scores = model(input_ids=inputs, attention_mask=inputs.ne(PAD).long(), use_cache=False).logits
            loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            seen += min(valid, remaining)
            steps += 1
            if steps == 1 or steps % 100 == 0 or seen == target:
                print(f"  {seen:,}/{target:,} training tokens, loss {loss.item():.4f}", flush=True)
            if seen == target:
                break
    return {"tokens": seen, "steps": steps, "training_seconds": perf_counter()-start,
            "parameters": sum(p.numel() for p in model.parameters())}


def ngram_counts(documents, orders, budget):
    """Count causal token patterns in a fixed training prefix budget"""
    counts = {n: Counter() for n in orders}
    for doc in documents:
        row = doc[:budget].tolist()
        for n, counts_for_order in counts.items():
            counts_for_order.update(zip(*(row[i:] for i in range(n))))
        budget -= len(row)
        if budget <= 0:
            break
    return counts


class ExpertFeatures:
    """Read local, recurrence and structure features from a causal prefix"""

    def __init__(self, tokenizer, counts):
        self.tokenizer, self.counts = tokenizer, counts
        self.spelling = [tokenizer.decode([i]) for i in range(tokenizer.get_vocab_size())]

    def __call__(self, prefix, candidates, log_probability):
        prior = Counter(prefix)
        last = {token:i for i,token in enumerate(prefix)}
        longest, followers = {}, Counter()
        for length in range(1, min(4, len(prefix)-1)+1):
            suffix = prefix[-length:]
            for i in range(len(prefix)-length):
                if prefix[i:i+length] == suffix:
                    token = prefix[i+length]
                    longest[token] = length
                    followers[token] += 1
        text = self.tokenizer.decode(prefix)
        stack = []
        for char in text:
            if char in DELIMITERS.values():
                stack.append(char)
            elif stack and DELIMITERS.get(char) == stack[-1]:
                stack.pop()
        mode = int(text.count("$") % 2) + 2*int(text.rfind(r"\cite{") > text.rfind("}"))
        result = {"local": [], "recurrence": [], "structure": []}
        for rank,(token,prob) in enumerate(zip(candidates.tolist(), log_probability.tolist(), strict=True)):
            search = [math.log1p(counts[(*prefix[-(n-1):],token)]) if len(prefix)>=n-1 else 0.
                      for n,counts in self.counts.items()]
            result["local"].append([prob, rank, *search])
            result["recurrence"].append([prob, prior[token], len(prefix)-last.get(token, -1),
                                         longest.get(token, 0), followers[token]])
            spelling = self.spelling[token]
            closes = bool(stack and spelling and DELIMITERS.get(spelling[0]) == stack[-1])
            identifier = bool(text and spelling and (text[-1].isalnum() or text[-1]=="_")
                              and (spelling[0].isalnum() or spelling[0]=="_"))
            result["structure"].append([prob,len(stack),closes,mode,identifier])
        return {key:np.asarray(value, dtype=np.float32) for key,value in result.items()}


@torch.inference_mode()
def predictions(model, data, device, size):
    """Yield next-token log probabilities on each physical window"""
    model.eval()
    for start in range(0,len(data),size):
        group = data[start:start+size]
        inputs, labels = batch(group,device)
        scores = model(input_ids=inputs, attention_mask=inputs.ne(PAD).long(), use_cache=False).logits
        yield group, labels, F.log_softmax(scores.float(),dim=-1)


def candidate_rows(data, shortlist, log_probability, features):
    """Create feature rows for one batch of shortlisted token predictions"""
    rows = {name:[] for name in ("local","recurrence","structure")}
    for i, sequence in enumerate(data):
        prefix = sequence[:-1].tolist()
        for j in range(len(prefix)):
            candidate = features(prefix[:j+1], shortlist[i,j], log_probability[i,j])
            for name,col in candidate.items():
                rows[name].append(col)
    return {name:np.stack(col) for name,col in rows.items()}


def collect(model, data, features, device, config, budget, seed):
    """Sample prefixes whose observed token is present in the shortlist"""
    cut = config["boosters"]["candidates"]["top_k"]
    order = np.random.default_rng(seed).permutation(len(data))
    rows = {name:[] for name in config["models"]["ssm_xgboost"]["experts"]}
    base, target = [], []
    count = 0
    for items, labels, logp in predictions(model,[data[i] for i in order],device,config["training"]["batch_size"]):
        log, ids = logp.topk(cut, dim=-1)
        log, ids, labels = log.cpu().numpy(), ids.cpu().numpy(), labels.cpu().numpy()
        for i, sequence in enumerate(items):
            prefix = sequence[:-1].tolist()
            for j in range(len(prefix)):
                hits = np.flatnonzero(ids[i,j] == labels[i,j])
                if not len(hits):
                    continue
                values = features(prefix[:j+1],ids[i,j],log[i,j])
                for name in rows:
                    rows[name].append(values[name])
                base.append(log[i,j])
                target.append(hits[0])
                count += 1
                if count >= budget:
                    break
            if count >= budget:
                break
        if count >= budget:
            break
    if not target:
        raise ValueError("No observed token enters the shortlist. Train the backbone longer or increase top_k.")
    return np.stack(base), np.asarray(target), {name:np.stack(value) for name,value in rows.items()}


def fit_experts(model, data, features, config, device, output, seed):
    """Fit independent binary trees and calibrate the shortlist mixture"""
    params = dict(config["boosters"]["xgboost"])
    rounds = params.pop("boosting_rounds")
    params.update(nthread=max(1, min(6, torch.get_num_threads())), seed=seed)
    cut = config["boosters"]["candidates"]["top_k"]
    _, target, rows = collect(model,data[config["boosters"]["fit_split"]],features,device,config,
                              config["boosters"]["max_fit_prefixes"],seed)
    labels = (np.arange(cut)[None,:] == target[:,None]).ravel().astype(np.float32)
    experts = {}
    for name, table in rows.items():
        matrix = xgb.QuantileDMatrix(table.reshape(-1,table.shape[-1]),label=labels,
                                    max_bin=params["max_bin"], nthread=params["nthread"])
        experts[name] = xgb.train(params,matrix,num_boost_round=rounds)
        experts[name].save_model(output/f"{name}.ubj")
    fit_prefixes = len(target)
    del rows, labels, matrix
    base, target, rows = collect(model,data[config["blend"]["fit_split"]],features,device,config,
                                 config["blend"]["max_fit_prefixes"],seed)
    margins = expert_margins(experts, rows).astype(np.float64)
    true = np.arange(len(target)), target
    def objective(weights):
        score = base + margins @ weights
        loss = logsumexp(score,axis=1)-score[true]
        gradient = (softmax(score,axis=1)[...,None]*margins).sum(axis=1)-margins[true]
        return loss.mean(), gradient.mean(axis=0)
    bounds = tuple(config["blend"]["expert_weight_bounds"])
    result = minimize(objective, np.zeros(len(experts)), jac=True, bounds=[bounds]*len(experts),method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Blend calibration failed: {result.message}")
    return experts, result.x, {"booster_prefixes":fit_prefixes, "blend_prefixes":len(target),
                               "weights":dict(zip(experts,result.x.tolist(),strict=True))}


def evaluate_tokens(model, data, settings, device, feature_extractor, mixture=None):
    """Measure token accuracy and full-vocabulary perplexity"""
    cut = settings["boosters"]["candidates"]["top_k"]
    nll, correct, count, recall = 0., 0, 0, 0
    for items, labels, logp in predictions(model,data,device,settings["training"]["batch_size"]):
        valid = labels != -100
        scores, ids = logp.topk(cut,dim=-1)
        truth = labels[valid]
        if mixture is not None:
            rows = candidate_rows(items, ids.cpu().numpy(), scores.cpu().numpy(), feature_extractor)
        recall += int((ids[valid] == truth[:,None]).any(dim=-1).sum())
        logp, ids, scores = logp[valid], ids[valid], scores[valid]
        if mixture is not None:
            experts, weights = mixture
            logp = adjust(logp, ids, scores, expert_margins(experts,rows) @ weights)
        correct += int((logp.argmax(-1)==truth).sum())
        nll -= float(logp.gather(-1,truth[:,None]).sum())
        count += len(truth)
    return {"scored_tokens":count,"next_token_perplexity":math.exp(nll/count),
            "next_token_top1_accuracy":correct/count, f"candidate_target_recall_at_{cut}":recall/count}


def completion_cases(documents, settings):
    """Read held-out equation and citation spans in source order"""
    cases = {}
    for kind in settings["prompt_spans"]:
        cases[kind] = list(islice(
            ((text[:m.start(1)], m[1]) for text in documents for m in COMPLETIONS[kind].finditer(text)),
            settings["max_examples_per_kind"],
        ))
    return cases


@torch.inference_mode()
def generate(model, prompt, stop, tokenizer, settings, device, features, mixture):
    """Greedily complete one prompt and time prefill and later tokens"""
    context = settings["data"]["context_tokens"]
    limit = settings["evaluation"]["generation"]["max_new_tokens"]
    cut = settings["boosters"]["candidates"]["top_k"]
    prefix = [BOS, *tokenizer.encode(prompt).ids]
    tokens = []
    prefill, generation = 0., 0.
    for step in range(limit):
        start = perf_counter()
        history = (prefix+tokens)[-context:]
        inputs = torch.tensor([history],device=device)
        logp = F.log_softmax(model(input_ids=inputs,use_cache=False).logits[0,-1].float(),dim=-1)
        if mixture is not None:
            experts, weights = mixture
            scores, ids = logp.topk(cut)
            rows = features(history,ids.cpu().numpy(),scores.cpu().numpy())
            logp = adjust(logp,ids,scores,expert_margins(experts,rows) @ weights)
        token = int(logp.argmax())
        elapsed = perf_counter()-start
        if step == 0:
            prefill = elapsed
        else:
            generation += elapsed
        if token == EOS:
            break
        tokens.append(token)
        text = tokenizer.decode(tokens)
        end = text.find(stop)
        if end >= 0:
            return text[:end+1], prefill, generation, max(0,len(tokens)-1)
    return tokenizer.decode(tokens), prefill, generation, max(0,len(tokens)-1)


def evaluate(model, test, cases, tokenizer, config, device, features, mixture=None):
    """Score held-out text and report single-prompt generation cost"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = evaluate_tokens(model,test,config,device,features,mixture)
    prefill, generated, elapsed = [], 0, 0.
    for kind, pairs in cases.items():
        correct = 0
        for prompt, target in pairs:
            text, first, seconds, count = generate(model,prompt,target[-1],tokenizer,config,device,features,mixture)
            correct += text == target
            prefill.append(first)
            generated += count
            elapsed += seconds
        result[f"{kind}_completion_examples"] = len(pairs)
        result[f"{kind}_completion_exact_match"] = correct/len(pairs) if pairs else None
    result["prefill_latency_ms"] = 1000*float(np.median(prefill)) if prefill else None
    result["generated_tokens_per_second"] = generated/elapsed if elapsed > 0 else None
    result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result


def smoke_data(config):
    """Make a small deterministic corpus to check the full benchmark on CPU"""
    config["data"]["tokenizer"]["vocabulary_size"] = 320
    config["data"]["context_tokens"] = 32
    config["models"]["transformer"].update(width=64,layers=2,heads=4)
    config["models"]["ssm"].update(width=64,layers=3,head_dim=32,state_size=16,chunk_size=16)
    config["training"].update(parameters_per_backbone=None,training_tokens_per_backbone=2048,
                              batch_size=4,seeds=[101])
    config["boosters"].update(max_fit_prefixes=128)
    config["boosters"]["candidates"]["top_k"] = 32
    config["boosters"]["xgboost"].update(boosting_rounds=8,max_depth=3,min_child_weight=2)
    config["blend"]["max_fit_prefixes"] = 128
    config["evaluation"]["generation"].update(max_examples_per_kind=2,max_new_tokens=12)
    return {
        split: [f"Detector sample {split} {i}: the measured mass is $m=M_i$. "
                f"For mass calibration use the measured mass $m=M_i$ and \\cite{{ref{i}}}."
                for i in range(12)]
        for split in config["data"]["files"]
    }


def run(config, output, device, smoke, seed):
    """Train the three benchmark variants and save the report and checkpoints"""
    documents = smoke_data(config) if smoke else read_documents(config["data"]["files"])
    tokenizer, tokens = tokenize(documents,config["data"]["tokenizer"])
    tokenizer.save(str(output/"tokenizer.json"))
    vocabulary = tokenizer.get_vocab_size()
    if not 1 < config["boosters"]["candidates"]["top_k"] < vocabulary:
        raise ValueError("Set top_k above 1 and below the fitted vocabulary size")
    local = config["boosters"]["experts"]["local"]
    counts = ngram_counts(tokens[local["count_split"]], local["ngram_orders"], local["max_count_tokens"])
    count_tables = {}
    for order, table in counts.items():
        count_tables[f"tokens_{order}"] = np.asarray(list(table), dtype=np.int32).reshape(-1, order)
        count_tables[f"counts_{order}"] = np.asarray(list(table.values()), dtype=np.int64)
    np.savez_compressed(output/"ngram_counts.npz", **count_tables)
    features = ExpertFeatures(tokenizer, counts)
    context = config["data"]["context_tokens"]
    datasets = {split:windows(rows,context) for split,rows in tokens.items()}
    training = datasets[config["training"]["train_split"]]
    test_split = config["evaluation"]["split"]
    cases = completion_cases(documents[test_split],config["evaluation"]["generation"])
    report = {"device":str(device),"precision":"float32","smoke":smoke,"vocabulary_size":vocabulary,
              "generation_recomputes_prefix":True,
              "document_counts":{split:len(rows) for split,rows in documents.items()},
              "scored_tokens_by_split":{split:sum(len(row)-1 for row in rows) for split,rows in datasets.items()},
              "results":{}}
    (output/"resolved.yml").write_text(yaml.safe_dump(config,sort_keys=False))
    seeds = [seed] if seed is not None else config["training"]["seeds"]
    for value in seeds:
        folder = output/f"seed-{value}"
        folder.mkdir()
        result = {}
        for name in ("transformer","ssm"):
            print(f"Training {name}, seed {value}",flush=True)
            torch.manual_seed(value)
            model = backbone(config,name,vocabulary,context).to(device)
            training_stats = train(model,training,config,device,value)
            model_folder = folder/name
            model.config.save_pretrained(model_folder)
            torch.save(model.state_dict(),model_folder/"checkpoint.pt")
            if name == "ssm":
                model.requires_grad_(False)
                start = perf_counter()
                print("Fitting three XGBoost specialists",flush=True)
                experts, weights, blend = fit_experts(model,datasets,features,config,device,folder,value)
                blend["training_seconds"] = perf_counter()-start
                (folder/"blend.json").write_text(json.dumps(blend,indent=2)+"\n")
            result[name] = {"training":training_stats,
                            "test":evaluate(model,datasets[test_split],cases,tokenizer,config,device,features)}
            if name == "ssm":
                result["ssm_xgboost"] = {
                    "training":{"shared_backbone":"ssm",**blend},
                    "test":evaluate(model,datasets[test_split],cases,tokenizer,config,device,features,(experts,weights)),
                }
            del model
        report["results"][str(value)] = result
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Results saved to {output/'report.json'}",flush=True)


def main():
    """Run the YAML benchmark or a small offline verification"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("ssm_xgboost_benchmark.yml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--seed", type=int, help="Run only this seed")
    parser.add_argument("--smoke", action="store_true", help="Use a small synthetic corpus and reduced training budget")
    args = parser.parse_args()
    mode = args.device
    if mode == "auto":
        mode = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    if mode == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.smoke:
        torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text())
    if not args.smoke:
        files = config["data"]["files"]
        for split, file in files.items():
            files[split] = str(args.config.resolve().parent / file)
        missing = [file for file in files.values() if not Path(file).is_file()]
        if missing:
            parser.error("Missing JSONL document files: "+", ".join(missing))
    args.output.mkdir(parents=True,exist_ok=False)
    run(config,args.output,torch.device(mode),args.smoke,args.seed)


if __name__ == "__main__":
    main()
