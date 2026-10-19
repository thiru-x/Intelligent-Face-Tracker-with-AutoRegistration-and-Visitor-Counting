# Safe sample outputs

These files are **synthetic examples**, prepared to demonstrate the output
formats without publishing visitor biometrics. The database IDs, audit events,
timestamps, media times, and image cards are illustrative; they are not an
export of the local visitor database, logs, video, or face crops.

- `visitor_sample.sqlite` contains a schema-compatible example identity, a
  synthetic 512-dimensional placeholder vector, entry/exit events, and audit
  rows. The vector is not a real face embedding and must not be used for
  recognition or copied into the application's live database.
- `events_sample.jsonl` shows representative structured audit records. It
  contains no embedding values.
- `images/` contains clearly marked SVG placeholders, not captured or
  anonymized visitor photographs.

The actual local database, logs, and dated face crops are excluded from Git
because they may contain sensitive biometric data. Only add real sample
outputs to a public repository when you have permission and have applied an
appropriate privacy review.

To inspect the sample database with Python:

```powershell
python -c "import sqlite3; c=sqlite3.connect('sample_outputs/visitor_sample.sqlite'); print('Faces:', c.execute('select count(*) from faces').fetchone()[0]); print('Events:', c.execute('select event_type, count(*) from events group by event_type').fetchall()); c.close()"
```
