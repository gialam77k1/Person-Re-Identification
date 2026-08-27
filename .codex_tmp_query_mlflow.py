import sqlite3
from datetime import datetime

db = r"E:\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\mlruns\mlflow.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

print('EXPERIMENTS')
for row in cur.execute("SELECT experiment_id, name FROM experiments ORDER BY experiment_id"):
    print(row)

print('\nTOP METRICS')
query = """
SELECT m.run_uuid, m.key, m.value, m.step, r.name, r.experiment_id, r.start_time
FROM metrics m
LEFT JOIN runs r ON m.run_uuid = r.run_uuid
WHERE lower(m.key) IN ('rank1','rank1_base','rank1_rerank','rank5','rank10','map','map_base','map_rerank')
ORDER BY m.value DESC
LIMIT 120
"""
for row in cur.execute(query):
    print(row)

print('\nRUNS LATEST')
for row in cur.execute("SELECT run_uuid, name, experiment_id, start_time, end_time, status FROM runs ORDER BY start_time DESC LIMIT 40"):
    print(row)
