"""Persistent causal feature population and delayed public-index training labels.

Every complete half-hour observation belongs here, including observations whose
model action was NONE. Pending outcomes never enter a training frame. This
module performs no network requests, loads no serialized model, and places no
orders.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from .features import FEATURES, SYMBOLS

MINUTE_MS=60_000
DAY_MS=86_400_000
LABEL_DELAY_MS=5_000
ROW_COLUMNS=FEATURES+["signal_ms","settlement_ms","y","is_flat"]


def _ms(value,label="timestamp"):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
            or not math.isfinite(value) or value!=int(value) or value<=0):
        raise ValueError(f"Invalid {label}")
    return int(value)


def _symbol_signal(symbol,signal_ms):
    if symbol not in SYMBOLS:
        raise ValueError("Feature history only supports BTCUSDT and ETHUSDT")
    stamp=_ms(signal_ms,"signal timestamp")
    if stamp%(30*MINUTE_MS):
        raise ValueError("Signal timestamp must be a half-hour boundary")
    return stamp


def _vector(features):
    if not isinstance(features,dict) or set(features)!=set(FEATURES):
        raise ValueError("Feature history requires exactly the frozen 60 historical inputs")
    values=[]
    for name in FEATURES:
        value=features[name]
        if isinstance(value,(bool,np.bool_)):
            raise ValueError("Feature values must be finite numbers")
        try:
            value=float(value)
        except (ValueError,TypeError):
            raise ValueError("Feature values must be finite numbers") from None
        if not math.isfinite(value):
            raise ValueError("Feature values must be finite numbers")
        values.append(value)
    return values


def _same_features(encoded,values):
    old=json.loads(encoded)
    return len(old)==len(values) and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(old,values))


class FeatureHistory:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        contract=json.dumps(FEATURES,separators=(",",":"))
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS feature_history_meta (
                  name TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feature_history (
                  symbol TEXT NOT NULL, signal_ms INTEGER NOT NULL,
                  settlement_ms INTEGER NOT NULL, features_json TEXT NOT NULL,
                  y INTEGER, is_flat INTEGER, entry_price REAL, exit_price REAL,
                  label_source TEXT, PRIMARY KEY(symbol,signal_ms)
                );
                CREATE INDEX IF NOT EXISTS feature_due ON feature_history(y,settlement_ms);
                CREATE TABLE IF NOT EXISTS feature_seed_imports (
                  symbol TEXT NOT NULL, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
                  training_cutoff_ms INTEGER NOT NULL, PRIMARY KEY(symbol,sha256)
                );
            """)
            db.execute("INSERT OR IGNORE INTO feature_history_meta VALUES('features',?)",(contract,))
            existing=db.execute("SELECT value FROM feature_history_meta WHERE name='features'").fetchone()[0]
            if existing!=contract:
                raise ValueError("Stored feature history has a different feature contract")

    @contextmanager
    def _db(self):
        connection=sqlite3.connect(self.path,timeout=5,isolation_level=None)
        connection.row_factory=sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
        finally:
            connection.close()

    def record_features(self,symbol,signal_ms,features):
        stamp=_symbol_signal(symbol,signal_ms)
        values=_vector(features)
        encoded=json.dumps(values,separators=(",",":"),allow_nan=False)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old=db.execute("SELECT features_json FROM feature_history WHERE symbol=? AND signal_ms=?",(symbol,stamp)).fetchone()
            if old is not None:
                if not _same_features(old["features_json"],values):
                    raise ValueError("Conflicting historical feature values; the original row was preserved")
                db.commit()
                return False
            db.execute("INSERT INTO feature_history(symbol,signal_ms,settlement_ms,features_json) VALUES(?,?,?,?)",
                       (symbol,stamp,stamp+31*MINUTE_MS,encoded))
            db.commit()
        return True

    def pending_labels(self,now_ms,limit=8):
        now=_ms(now_ms,"observation timestamp")
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=1000:
            raise ValueError("Pending label limit must be between 1 and 1000")
        with self._db() as db:
            rows=db.execute("""SELECT symbol,signal_ms,settlement_ms FROM feature_history
                WHERE y IS NULL AND settlement_ms+?<=? ORDER BY signal_ms,symbol LIMIT ?""",
                            (LABEL_DELAY_MS,now,limit)).fetchall()
        return [{"symbol":row["symbol"],"signal_ms":row["signal_ms"],
                 "entry_ms":row["signal_ms"]+MINUTE_MS,"settlement_ms":row["settlement_ms"]} for row in rows]

    def update_label(self,symbol,signal_ms,entry_price,exit_price):
        stamp=_symbol_signal(symbol,signal_ms)
        values=[]
        for raw in (entry_price,exit_price):
            if isinstance(raw,(bool,np.bool_)):
                raise ValueError("Index execution observations must be positive finite prices")
            try:
                number=float(raw)
            except (TypeError,ValueError):
                raise ValueError("Index execution observations must be positive finite prices") from None
            if not math.isfinite(number) or number<=0:
                raise ValueError("Index execution observations must be positive finite prices")
            values.append(number)
        entry,exit_value=values
        y,flat=int(exit_value>entry),int(exit_value==entry)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old=db.execute("SELECT * FROM feature_history WHERE symbol=? AND signal_ms=?",(symbol,stamp)).fetchone()
            if old is None:
                raise KeyError("Cannot label an absent historical feature observation")
            if old["y"] is not None:
                if old["y"]!=y or old["is_flat"]!=flat:
                    raise ValueError("Conflicting historical index outcome; original label preserved")
                if old["entry_price"] is not None and (not math.isclose(old["entry_price"],entry,rel_tol=0,abs_tol=1e-8)
                        or not math.isclose(old["exit_price"],exit_value,rel_tol=0,abs_tol=1e-8)):
                    raise ValueError("Conflicting historical execution price; original observations preserved")
                if old["entry_price"] is not None:
                    db.commit()
                    return False
            db.execute("""UPDATE feature_history SET y=?,is_flat=?,entry_price=?,exit_price=?,
                label_source='official_index_observation' WHERE symbol=? AND signal_ms=?""",
                       (y,flat,entry,exit_value,symbol,stamp))
            db.commit()
        return True

    def training_rows(self,cutoff_ms,days=180):
        cutoff=_ms(cutoff_ms,"training cutoff")
        if isinstance(days,bool) or not isinstance(days,int) or not 1<=days<=2000:
            raise ValueError("Training days must be between 1 and 2000")
        output={}
        with self._db() as db:
            for symbol in SYMBOLS:
                rows=db.execute("""SELECT * FROM feature_history WHERE symbol=? AND signal_ms>=?
                    AND signal_ms<? AND settlement_ms<? AND y IS NOT NULL
                    ORDER BY signal_ms""",(symbol,cutoff-days*DAY_MS,cutoff,cutoff)).fetchall()
                records=[json.loads(row["features_json"])+[row["signal_ms"],row["settlement_ms"],row["y"],bool(row["is_flat"])] for row in rows]
                frame=pd.DataFrame(records,columns=ROW_COLUMNS)
                frame=frame.astype({"signal_ms":"int64","settlement_ms":"int64","y":"int8","is_flat":"bool"})
                frame.attrs["features"]=FEATURES.copy()
                output[symbol]=frame
        return output

    def import_seed(self,bundle_dir,manifest):
        """Import only hash-verified CSV observations, never joblib/model objects.

        Reimporting an identical seed is a no-op. Overlapping seeds cannot
        replace earlier recorded features/outcomes, including rows collected by
        the live worker since its initial bundle was installed.
        """
        directory=Path(bundle_dir).resolve()
        if not isinstance(manifest,dict) or manifest.get("features")!=FEATURES:
            raise ValueError("Seed manifest has a different feature contract")
        cutoff=_ms(manifest.get("training_cutoff_ms"),"seed training cutoff")
        if set(manifest.get("models",{}))!=set(SYMBOLS):
            raise ValueError("Seed manifest must include exactly BTCUSDT and ETHUSDT")
        result={}
        prepared=[]
        for symbol in SYMBOLS:
            descriptor=manifest["models"][symbol]
            name=descriptor.get("training_samples_file")
            sha=descriptor.get("training_samples_sha256")
            if (not isinstance(name,str) or not name or "\\" in name or ":" in name
                    or not isinstance(sha,str) or re.fullmatch(r"[0-9a-fA-F]{64}",sha) is None):
                raise ValueError("Seed manifest requires a safe CSV filename and SHA-256")
            path=(directory/name).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or not name.endswith(".csv.gz"):
                raise ValueError("Seed CSV must stay inside the verified model bundle")
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=sha.lower():
                raise ValueError("Training seed checksum mismatch")
            with self._db() as db:
                prior=db.execute("SELECT row_count FROM feature_seed_imports WHERE symbol=? AND sha256=?",(symbol,actual)).fetchone()
            if prior is not None:
                result[symbol]={"inserted":0,"already_imported":True,"seed_rows":prior["row_count"]}
                continue
            frame=pd.read_csv(path,float_precision="round_trip")
            if list(frame.columns)!=ROW_COLUMNS or frame.empty:
                raise ValueError("Training seed CSV schema or population is invalid")
            matrix=frame[FEATURES].to_numpy(dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError("Nonfinite training seed features")
            times=frame.signal_ms.to_numpy()
            settles=frame.settlement_ms.to_numpy()
            if (not np.isfinite(times).all() or np.any(times!=times.astype(np.int64))
                    or np.any(times<=0) or np.any(times%(30*MINUTE_MS))
                    or np.any(np.diff(times)<=0) or np.any(settles!=times+31*MINUTE_MS)
                    or np.any(settles>=cutoff)):
                raise ValueError("Seed timestamps must be unique, settled, and before its cutoff")
            if not frame.y.isin([0,1]).all() or not frame.is_flat.isin([True,False]).all():
                raise ValueError("Invalid training seed outcomes")
            if ((frame.is_flat.astype(bool))&(frame.y!=0)).any():
                raise ValueError("A flat index observation cannot have an up label")
            prepared.append((symbol,actual,frame))
        # Validate every input file before adding any previously unseen seed.
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for symbol,sha,frame in prepared:
                # Another worker may have imported it since the file read.
                prior=db.execute("SELECT row_count FROM feature_seed_imports WHERE symbol=? AND sha256=?",(symbol,sha)).fetchone()
                if prior is not None:
                    result[symbol]={"inserted":0,"already_imported":True,"seed_rows":prior["row_count"]}
                    continue
                inserted=0
                for row in frame.itertuples(index=False,name=None):
                    vector=[float(value) for value in row[:len(FEATURES)]]
                    signal,settlement,y,flat=row[len(FEATURES):]
                    old=db.execute("SELECT features_json,y,is_flat FROM feature_history WHERE symbol=? AND signal_ms=?",(symbol,int(signal))).fetchone()
                    if old is not None:
                        if not _same_features(old["features_json"],vector):
                            raise ValueError("Training seed conflicts with recorded historical features")
                        if old["y"] is not None and (old["y"]!=int(y) or old["is_flat"]!=int(flat)):
                            raise ValueError("Training seed conflicts with a recorded historical label")
                        if old["y"] is None:
                            db.execute("UPDATE feature_history SET y=?,is_flat=?,label_source='verified_training_seed' WHERE symbol=? AND signal_ms=?",
                                       (int(y),int(flat),symbol,int(signal)))
                        continue
                    db.execute("""INSERT INTO feature_history(symbol,signal_ms,settlement_ms,features_json,y,is_flat,label_source)
                        VALUES(?,?,?,?,?,?,'verified_training_seed')""",
                               (symbol,int(signal),int(settlement),json.dumps(vector,separators=(",",":"),allow_nan=False),int(y),int(flat)))
                    inserted+=1
                db.execute("INSERT INTO feature_seed_imports VALUES(?,?,?,?)",(symbol,sha,len(frame),cutoff))
                result[symbol]={"inserted":inserted,"already_imported":False,"seed_rows":len(frame)}
            db.commit()
        return result

    def status(self):
        with self._db() as db:
            output={}
            for symbol in SYMBOLS:
                row=db.execute("""SELECT COUNT(*),SUM(y IS NULL),SUM(y IS NOT NULL),MAX(signal_ms),
                    MAX(CASE WHEN y IS NOT NULL THEN settlement_ms END) FROM feature_history WHERE symbol=?""",(symbol,)).fetchone()
                output[symbol]={"rows":row[0],"pending":row[1] or 0,"labeled":row[2] or 0,
                                "latest_signal_ms":row[3],"latest_labeled_settlement_ms":row[4]}
        return {"symbols":output,"total_rows":sum(row["rows"] for row in output.values()),
                "pending_labels":sum(row["pending"] for row in output.values())}
