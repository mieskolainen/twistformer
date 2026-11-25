# Cumulative event and object selection yields
#
# mikael.mieskolainen@cern.ch (2026)


class Cutflow:
    def __init__(self, stages, *, groups=(), grouped_stages=()):
        """
        Book ordered selection stages with optional group yields

        Args:
            stages:  Mapping from stage keys to printed cut descriptions in selection order
            groups:  Group names in column order
            grouped_stages:  Stages where the group is known
        """
        self.labels = dict(stages)
        self.counts = dict.fromkeys(self.labels, 0)
        self.groups = tuple(groups)
        self.group_counts = {stage: dict.fromkeys(self.groups, 0) for stage in grouped_stages}

    def add(self, stage, count=1, *, group=None):
        """
        Count objects that pass this stage and all preceding cuts

        Args:
            stage:  Booked stage key
            count:  Number of passing objects
            group:  Group name for a stage with a known classification
        """
        if stage in self.group_counts:
            self.group_counts[stage][group] += count
        self.counts[stage] += count

    def merge(self, other):
        """
        Combine yields from disjoint input ranges

        Args:
            other:  Cutflow with the same ordered stages and groups
        """
        if (list(self.labels.items()) != list(other.labels.items()) or self.groups != other.groups
                or self.group_counts.keys() != other.group_counts.keys()):
            raise ValueError("Cutflow stages and groups must match")
        for stage, count in other.counts.items():
            self.counts[stage] += count
        for stage, groups in other.group_counts.items():
            for group, count in groups.items():
                self.group_counts[stage][group] += count

    def table(self, title):
        """
        Format raw yields and efficiencies relative to the previous and first stages

        Args:
            title:  Population counted in the table
        Returns:
            text:  Aligned selection table
        """
        header = ("Selection", "Passed", "Removed", "Step %", "Total %", *self.groups)
        rows = [header]
        initial = next(iter(self.counts.values()), 0)
        previous = initial
        for index, (stage, count) in enumerate(self.counts.items()):
            groups = self.group_counts.get(stage, {})
            rows.append((self.labels[stage], f"{count:,}", f"{previous - count:,}" if index else "-",
                         f"{100 * count / previous:.2f}" if previous else "-",
                         f"{100 * count / initial:.2f}" if initial else "-",
                         *(f"{groups[group]:,}" if group in groups else "-" for group in self.groups)))
            previous = count
        widths = [max(map(len, column)) for column in zip(*rows)]
        lines = [title]
        for row in rows:
            numbers = [value.rjust(width) for value, width in zip(row[1:], widths[1:], strict=True)]
            lines.append("  ".join([row[0].ljust(widths[0]), *numbers]))
        return "\n".join(lines)
