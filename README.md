# WhitePrintAudioEngine — Audition

9次元 Time-Series Circuit Envelope Function.

Audio → Physics JSON. Forget everything after response.

## API (Internal)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/internal/analyze` | Multipart audio → analysis JSON |
| POST | `/internal/analyze-url` | Audio URL → analysis JSON |
| GET | `/health` | Liveness probe |

## Deploy

```bash
gcloud run deploy aimastering-audition \
  --source . --region asia-northeast1 \
  --memory 2Gi --cpu 2 --concurrency 1 --ingress internal
```

© YOMIBITO SHIRAZU — WhitePrintAudioEngine
