import pytest

from atlas.data import load_candles


def test_load_mt5_export_tab_separated(tmp_path):
    f = tmp_path / "xau.csv"
    f.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t01:00:00\t2060.5\t2062.0\t2059.0\t2061.0\t850\t0\t25\n"
        "2024.01.02\t02:00:00\t2061.0\t2063.5\t2060.5\t2063.0\t900\t0\t25\n",
        encoding="utf-8",
    )
    df = load_candles(f)
    assert len(df) == 2
    assert df.index[0].isoformat() == "2024-01-02T01:00:00"
    assert df["open"].iloc[0] == 2060.5
    assert df["close"].iloc[1] == 2063.0
    assert df["volume"].iloc[0] == 850


def test_load_simple_comma_csv(tmp_path):
    f = tmp_path / "xau.csv"
    f.write_text(
        "time,open,high,low,close,volume\n"
        "2024-01-02 01:00:00,2060.5,2062,2059,2061,10\n",
        encoding="utf-8",
    )
    df = load_candles(f)
    assert len(df) == 1
    assert df["high"].iloc[0] == 2062


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candles(tmp_path / "nope.csv")


def test_missing_columns_raises(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("time,open,close\n2024-01-02,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_candles(f)


def test_loader_does_not_silently_fix_duplicates(tmp_path):
    f = tmp_path / "dup.csv"
    f.write_text(
        "time,open,high,low,close\n"
        "2024-01-02 01:00:00,1,2,0.5,1.5\n"
        "2024-01-02 01:00:00,1,2,0.5,1.5\n",
        encoding="utf-8",
    )
    assert len(load_candles(f)) == 2


MT5_TEXT = (
    "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
    "2024.01.02\t01:00:00\t2060.5\t2062.0\t2059.0\t2061.0\t850\t0\t25\n"
    "2024.01.02\t02:00:00\t2061.0\t2063.5\t2060.5\t2063.0\t900\t0\t25\n"
)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-8-sig"])
def test_load_mt5_export_in_other_encodings(tmp_path, encoding):
    f = tmp_path / "xau.csv"
    if encoding == "utf-16-le":  # UTF-16 little endian AVEC BOM, comme Windows
        f.write_bytes(b"\xff\xfe" + MT5_TEXT.encode("utf-16-le"))
    else:
        f.write_bytes(MT5_TEXT.encode(encoding))
    df = load_candles(f)
    assert len(df) == 2
    assert df["close"].iloc[1] == 2063.0
    assert df["volume"].iloc[0] == 850
