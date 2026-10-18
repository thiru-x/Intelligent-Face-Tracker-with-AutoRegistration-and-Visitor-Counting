
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "visitors.db"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row

    faces = conn.execute(
        "SELECT face_id, embedding, embedding_dimension FROM faces"
    ).fetchall()

    events = conn.execute(
        "SELECT event_id, face_id, event_type, timestamp, image_path "
        "FROM events ORDER BY event_id DESC"
    ).fetchall()

print("\n=== FACE DATABASE ===")
print(f"Registered identities: {len(faces)}")

invalid_embeddings = []
for face in faces:
    if len(face["embedding"]) != face["embedding_dimension"] * 4:
        invalid_embeddings.append(face["face_id"])

print(f"Invalid embedding records: {len(invalid_embeddings)}")

print("\n=== EVENT DATABASE ===")
print(f"Total events: {len(events)}")

event_counts = Counter(event["event_type"] for event in events)
print(f"Entry events: {event_counts.get('entry', 0)}")
print(f"Exit events: {event_counts.get('exit', 0)}")

missing_images = []
unknown_faces = []
for event in events:
    image_path = event["image_path"]

    if image_path:
        image_file = Path(image_path)
        if not image_file.is_absolute():
            image_file = ROOT / image_file
        if not image_file.is_file():
            missing_images.append(event["event_id"])

    if not any(face["face_id"] == event["face_id"] for face in faces):
        unknown_faces.append(event["event_id"])

print(f"Events with missing image files: {len(missing_images)}")
print(f"Events referencing unknown identities: {len(unknown_faces)}")

print("\n=== MOST RECENT 10 EVENTS ===")
for event in events[:10]:
    print(
        f"{event['timestamp']} | {event['event_type'].upper():5} "
        f"| {event['face_id']} | image={event['image_path']}"
    )

print("\nAudit complete. Historical events may include previous test runs.")
