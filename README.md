# WhitePrintAudioEngine — Audition

Vertex AI Gemini による楽曲分析エンジン。音を聴き、全てを決める。

## 役割

パイプラインの最初のAI。トラックを聴いて以下を**全て自律的に決定**する：

| 出力 | 内容 |
|------|------|
| `estimated_bpm` | テンポ (BPM) |
| `estimated_key` | 調 (例: C major, Ab minor) |
| `genre` | ジャンル |
| `mood` | 雰囲気・エネルギー |
| `sections` | 楽曲構造 (Intro, Verse, Drop 等) |
| `constraints` | DSPパラメータ制約 (max/min/force) |
| `recommended_target_lufs` | **最適LUFS目標** — AIが曲ごとに判断 |
| `recommended_target_true_peak` | **最適True Peak天井** — AIが曲ごとに判断 |
| `track_title` / `track_title_romaji` | 曲名（入稿用） |

**ハードコードされたデフォルト値は一切存在しない。** 全ての数値はVertex AI Geminiがトラックの信号特性から判断する。

## 技術

- **Vertex AI Gemini** — 音声直接入力による分析
- **BS.1770-4** — K-weighting LUFS測定
- **9次元 Time-Series Circuit Envelope** — 信号の物理特性抽出

## API (Internal)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/internal/analyze` | Multipart audio → analysis JSON |
| POST | `/internal/analyze-url` | Audio URL → analysis JSON |
| GET | `/health` | Liveness probe |

## Deploy

```bash
gcloud run deploy whiteprintaudioengine-audition \
  --source . --region asia-northeast1 \
  --memory 4Gi --cpu 2 --timeout 300 --concurrency 1 --ingress internal
```

© YOMIBITO SHIRAZU — WhitePrintAudioEngine
