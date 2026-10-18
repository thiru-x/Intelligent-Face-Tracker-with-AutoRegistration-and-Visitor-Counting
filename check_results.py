
import sqlite3
from pathlib import Path

db_path = Path("data/visitors.db")

if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

with sqlite3.connect(db_path) as conn:
    print("\n=== DATABASE SUMMARY ===")

    total_faces = conn.execute(
        "SELECT COUNT(*) FROM faces"
    ).fetchone()[0]

    total_events = conn.execute(
        "SELECT COUNT(*) FROM events"
    ).fetchone()[0]

    print(f"Registered identities: {total_faces}")
    print(f"Total events: {total_events}")

    print("\n=== EVENTS BY TYPE ===")
    for event_type, count in conn.execute(
        "SELECT event_type, COUNT(*) FROM events "
        "GROUP BY event_type ORDER BY event_type"
    ):
        print(f"{event_type}: {count}")

    print("\n=== IDENTITIES WITH REPEATED ENTRIES ===")
    rows = conn.execute("""
        SELECT face_id, COUNT(*) AS entries
        FROM events
        WHERE event_type = 'entry'
        GROUP BY face_id
        HAVING COUNT(*) > 1
        ORDER BY entries DESC
        LIMIT 10
    """).fetchall()

    if rows:
        for face_id, entries in rows:
            print(f"{face_id}: {entries} entries")
    else:
        print("No identities with repeated entry events.")
