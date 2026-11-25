# HDF5 top-tag column compatibility without pandas or PyTables
#
# mikael.mieskolainen@cern.ch (2026)

import pickle
import sys
import subprocess

import numpy as np
import pytest

from drivers.toptag.reader import read_columns, _column_names, _read_rows

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize("layout", ["table", "native_names", "fixed"])
@pytest.mark.parametrize("limit", [-1, 4, 100])
def test_hdf_columns_without_legacy_dependencies(tmp_path, monkeypatch, layout, limit):
    """
    Preserve sampled numeric columns and source indices without legacy imports

    Args:
        tmp_path:  Temporary HDF5 directory
        monkeypatch:  Import isolation fixture
        layout:  Compound table or fixed numeric blocks
        limit:  Requested event count
    Returns:
        None: Column values and seeded source indices match the reference
    """
    monkeypatch.setitem(sys.modules, "tables", None)
    monkeypatch.setitem(sys.modules, "pandas", None)
    path = tmp_path / "jets.h5"
    index = np.arange(12) + 100
    momenta = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.arange(12) % 2
    with h5py.File(path, "w") as store:
        group = store.create_group("table")
        if layout == "fixed":
            group["axis1"] = index
            group["block0_items"] = np.array([b"E_0", b"PX_0"])
            group["block0_values"] = momenta
            group["block1_items"] = np.array([b"is_signal_new"])
            group["block1_values"] = labels[:, None]
        else:
            values = np.empty(
                12,
                dtype=[
                    ("index", "i8"),
                    ("values_block_0", "f4", (2,)),
                    ("is_signal_new", "i8"),
                ],
            )
            values["index"], values["values_block_0"], values["is_signal_new"] = (
                index,
                momenta,
                labels,
            )
            table = group.create_dataset("table", data=values, compression="gzip")
            table.attrs["values_block_0_kind"] = (
                np.bytes_(pickle.dumps(["E_0", "PX_0"], protocol=0))
                if layout == "table"
                else np.array([b"E_0", b"PX_0"])
            )
    columns, indices = read_columns(path, limit, 101)
    rows = (
        np.sort(np.random.default_rng(101).choice(12, limit, replace=False))
        if 0 < limit < 12
        else np.arange(12)
    )
    np.testing.assert_array_equal(indices, index[rows])
    np.testing.assert_array_equal(columns["E_0"], momenta[rows, 0])
    np.testing.assert_array_equal(columns["PX_0"], momenta[rows, 1])
    np.testing.assert_array_equal(columns["is_signal_new"], labels[rows])


@pytest.mark.parametrize("compression", ["Blosc", "Blosc2", "BZip2"])
@pytest.mark.parametrize("layout", ["table", "fixed"])
def test_compressed_reads_in_fresh_process(tmp_path, compression, layout):
    """
    Load compressed HDF5 through reader-owned plugin registration

    Args:
        tmp_path:  Temporary HDF5 directory
        compression:  Filter used to encode the event data
        layout:  Compound table or fixed numeric frame
    Returns:
        None: Full and sampled reads work without legacy HDF5 readers
    """
    plugin = pytest.importorskip("hdf5plugin")
    path = tmp_path / "compressed.h5"
    options = getattr(plugin, compression)()
    with h5py.File(path, "w") as store:
        group = store.create_group("table")
        if layout == "table":
            values = np.zeros(64, dtype=[("index", "i8"), ("E_0", "f4")])
            values["index"] = np.arange(64) + 100
            values["E_0"] = np.arange(64)
            group.create_dataset("table", data=values, **options)
        else:
            group.create_dataset("axis1", data=np.arange(64) + 100, **options)
            group["block0_items"] = np.array([b"E_0"])
            group.create_dataset(
                "block0_values",
                data=np.arange(64, dtype=np.float32)[:, None],
                **options,
            )
    script = """
import sys
import numpy as np
sys.modules['tables'] = None
sys.modules['pandas'] = None
from drivers.toptag.reader import read_columns
assert 'hdf5plugin' not in sys.modules
for limit in [-1, 5]:
    columns, indices = read_columns(sys.argv[1], limit, 101)
    rows = np.arange(64) if limit < 0 else np.sort(np.random.default_rng(101).choice(64, limit, replace=False))
    np.testing.assert_array_equal(indices, rows + 100)
    np.testing.assert_array_equal(columns['E_0'], rows)
assert 'hdf5plugin' in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_compression_package_has_install_hint(monkeypatch):
    """
    Report a missing compression dependency before opening the dataset

    Args:
        monkeypatch:  Import isolation fixture
    Returns:
        None: The error provides the installation command
    """
    monkeypatch.setitem(sys.modules, "hdf5plugin", None)
    with pytest.raises(ImportError, match="python -m pip install hdf5plugin"):
        read_columns("unused.h5", -1, 101)


def test_column_metadata_rejects_objects():
    """
    Reject serialized object reconstruction and malformed column labels

    Returns:
        None: Only plain label sequences are accepted
    """
    for value in (pickle.dumps(eval), b"not a pickle", pickle.dumps([42])):
        with pytest.raises(ValueError):
            _column_names(value)
    assert _column_names(pickle.dumps(["E_0", "PX_0"], protocol=4)) == ["E_0", "PX_0"]


def test_sampled_reads_use_bounded_blocks():
    """
    Keep sampled read buffers bounded for wide constituent tables

    Returns:
        None: Sparse source indices survive multiple bounded reads
    """

    class Dataset:
        shape = (20000, 1000)
        dtype = np.dtype("float32")

        def __getitem__(self, selection):
            """
            Supply synthetic rows while checking the contiguous read size

            Args:
                selection:  Contiguous source slice
            Returns:
                values: Repeated source indices across columns
            """
            assert isinstance(selection, slice)
            assert (selection.stop - selection.start) * 4000 <= 8 * 1024**2
            return np.broadcast_to(
                np.arange(selection.start, selection.stop)[:, None],
                (selection.stop - selection.start, 1000),
            )

    rows = np.array([0, 4000, 19000])
    np.testing.assert_array_equal(_read_rows(Dataset(), rows)[:, 0], rows)
    assert _read_rows(Dataset(), np.array([], dtype=int)).shape == (0, 1000)


def test_train_row_offset(tmp_path):
    """Reserve the constituent training prefix before event sampling"""
    path = tmp_path / "training.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("table")
        group["axis1"] = np.arange(30) + 1000
        group["block0_items"] = np.array([b"is_signal_new"])
        group["block0_values"] = np.arange(30)[:, None] % 2
    for limit in (-1, 5):
        columns, ids = read_columns(path, limit, 101, start=12)
        expected = np.arange(12, 30)
        if limit > 0:
            expected = np.sort(np.random.default_rng(101).choice(expected, limit, replace=False))
        np.testing.assert_array_equal(ids, 1000 + expected)
        np.testing.assert_array_equal(columns["is_signal_new"], expected % 2)
