import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=7))
db = r"E:\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\mlruns\mlflow.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

keys = ['rank1','rank5','rank10','mAP','mAP_base','mAP_rerank','rank1_base','rank1_rerank']
for key in keys:
    row = cur.execute(
        """
        SELECT m.run_uuid, m.key, m.value, m.step, r.name, r.experiment_id, r.start_time
        FROM metrics m
        LEFT JOIN runs r ON m.run_uuid = r.run_uuid
        WHERE m.key = ?
        ORDER BY m.value DESC
        LIMIT 1
        """,
        (key,)
    ).fetchone()
    print(key, row)

print('\nTOP RUNS BY MAX rank1')
rows = cur.execute(
    """
    SELECT m.run_uuid, MAX(m.value) as best_rank1, r.name, r.experiment_id, r.start_time
    FROM metrics m
    LEFT JOIN runs r ON m.run_uuid = r.run_uuid
    WHERE m.key = 'rank1'
    GROUP BY m.run_uuid
    ORDER BY best_rank1 DESC
    LIMIT 10
    """
).fetchall()
for row in rows:
    print(row)
