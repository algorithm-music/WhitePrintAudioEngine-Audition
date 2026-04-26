import requests, json
r = requests.post('https://audition.aimastering.tech/internal/analyze-url',
    json={'audio_url': 'https://storage.googleapis.com/aidriven-mastering-fyqu-source-bucket/analyze-temp/aki_no_ta_no.wav'},
    timeout=300)
d = r.json()
print('HTTP', r.status_code)
ti = d.get('track_identity', {})
print('BPM:', ti.get('bpm'), 'src:', ti.get('bpm_source'))
print('Key:', ti.get('key'), 'src:', ti.get('key_source'))
print('Genre:', ti.get('genre'))
print('Mood:', ti.get('mood'))
print('Problems:', json.dumps(d.get('detected_problems', []), indent=2, ensure_ascii=False))
print('Guardrails:', json.dumps(d.get('param_guardrails'), indent=2, ensure_ascii=False))
print('Target LUFS:', d.get('recommended_target_lufs'))
print('Target Peak:', d.get('recommended_target_true_peak'))
print('Title:', d.get('track_title'))
print('Romaji:', d.get('track_title_romaji'))
for s in d.get('physical_sections', []):
    ctx = s.get('semantic_context')
    print(f"  {s['section_id']} ({s['start_sec']}-{s['end_sec']}s) ctx={ctx}")