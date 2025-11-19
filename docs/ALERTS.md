# Sistema de Alertas de Segurança - VisionInference

## 📋 Visão Geral

O sistema detecta automaticamente violações de segurança (pessoas sem capacete) e:
1. ✅ Rastreia detecções consecutivas por mais de 5 segundos (configurável)
2. ✅ Salva automaticamente 10 segundos de vídeo do incidente
3. ✅ Envia notificação ao backend com detalhes da violação
4. ✅ Implementa cooldown de 30 segundos entre alertas da mesma fonte

## 🎯 Classes de Violação

O sistema monitora as seguintes classes:
- `head` - Cabeça sem capacete detectada
- `person_no_helmet` - Pessoa explicitamente sem capacete

## ⚙️ Configuração

### Variáveis de Ambiente (`.env`)

```properties
# Alertas
ENABLE_ALERTS=true                      # Ativar/desativar sistema de alertas
VIOLATION_DURATION_THRESHOLD=5.0        # Segundos de violação antes de alertar
VIOLATION_CONFIDENCE_THRESHOLD=0.75     # Confiança mínima (0.0-1.0)
VIDEO_BUFFER_SECONDS=10.0               # Segundos de vídeo no buffer
ALERT_COOLDOWN_SECONDS=30.0             # Cooldown entre alertas
BACKEND_API_URL=http://localhost:8080/api/inference
```

### Parâmetros do Pipeline

```python
from src.pipelines.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    sources=["0"],                       # Fontes de vídeo
    enable_alerts=True,                  # Ativar alertas
    violation_threshold_seconds=5.0,     # Threshold de violação
    violation_confidence=0.75,           # Confiança mínima
    video_buffer_seconds=10.0,           # Buffer de vídeo
    incidents_dir="incidents"            # Diretório para vídeos
)
```

## 📊 Fluxo de Detecção

```
Frame → Inference → AlertManager
                          ↓
                    Violação detectada?
                          ↓
                    Duração > 5s?
                          ↓
                    Confiança > 0.75?
                          ↓
                    Cooldown OK?
                          ↓
                    [Salvar Vídeo] → [Notificar Backend]
```

## 📹 Vídeos de Incidente

### Localização
Os vídeos são salvos automaticamente em:
```
incidents/violation_{source}_{timestamp}.mp4
```

### Exemplo
```
incidents/violation_0_20231119_153045.mp4
```

### Conteúdo
- 10 segundos de vídeo (5s antes + 5s durante a violação)
- Frames originais sem anotações
- Codec: MP4V
- FPS: 30 (ou FPS da fonte)

## 🔔 Notificações ao Backend

### Endpoint
```
POST {BACKEND_API_URL}
Content-Type: application/json
```

### Payload
```json
{
  "violation_id": "0_1700405445",
  "source": "0",
  "violation_type": "no_helmet",
  "start_time": "2023-11-19T15:30:45Z",
  "end_time": "2023-11-19T15:30:52Z",
  "duration_seconds": 7.2,
  "max_confidence": 0.89,
  "frame_count": 216,
  "video_path": "incidents/violation_0_20231119_153045.mp4"
}
```

### Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `violation_id` | string | ID único da violação |
| `source` | string | Fonte do vídeo |
| `violation_type` | string | Tipo de violação (no_helmet) |
| `start_time` | datetime | Início da violação (ISO 8601) |
| `end_time` | datetime | Fim da violação (ISO 8601) |
| `duration_seconds` | float | Duração total em segundos |
| `max_confidence` | float | Maior confiança detectada (0.0-1.0) |
| `frame_count` | int | Número de frames com violação |
| `video_path` | string | Caminho do vídeo salvo |

## 🔧 Uso Programático

### Monitoramento Manual

```python
from src.utils.alert_manager import AlertManager
from src.inference.detector import Detector

# Criar detector e alert manager
detector = Detector()
alert_manager = AlertManager(
    violation_duration_threshold=5.0,
    confidence_threshold=0.75
)

# Processar frames
for frame in video_stream:
    result = detector.predict(frame, source="camera_1")
    violation = alert_manager.process_result(result)
    
    if violation:
        print(f"⚠️ Violação detectada: {violation.violation_id}")
        alert_manager.send_notification(violation)
```

### Estatísticas

```python
stats = alert_manager.get_statistics()
print(f"Total de violações: {stats['total_violations']}")
print(f"Violações ativas: {stats['active_violations']}")
print(f"Fontes monitoradas: {stats['sources_tracked']}")
```

## 🐛 Troubleshooting

### Vídeos não estão sendo salvos
1. Verifique permissões do diretório `incidents/`
2. Confirme que `ENABLE_ALERTS=true` no `.env`
3. Verifique logs para erros do VideoBuffer

### Notificações não estão sendo enviadas
1. Verifique se o backend está rodando
2. Confirme `BACKEND_API_URL` no `.env`
3. Verifique conectividade de rede
4. Veja logs para erros de timeout/conexão

### Muitos alertas sendo gerados
1. Aumente `ALERT_COOLDOWN_SECONDS`
2. Aumente `VIOLATION_CONFIDENCE_THRESHOLD`
3. Aumente `VIOLATION_DURATION_THRESHOLD`

### Alertas não são acionados
1. Verifique se as classes detectadas são `head` ou `person_no_helmet`
2. Reduza `VIOLATION_CONFIDENCE_THRESHOLD` (< 0.75)
3. Reduza `VIOLATION_DURATION_THRESHOLD` (< 5.0)
4. Verifique logs DEBUG para ver detecções

## 📈 Performance

### Impacto no Desempenho
- **VideoBuffer**: ~50-100MB RAM por fonte (10s @ 30fps)
- **AlertManager**: Negligível (<1MB RAM)
- **Salvamento de vídeo**: ~2-3s por incidente (não bloqueia inferência)

### Otimizações
- Buffer de vídeo usa `deque` (O(1) para operações)
- Notificações com timeout de 5s
- Salvamento de vídeo após detecção (não em tempo real)

## 📝 Logs

### Exemplos

```
INFO: Violation tracking started for camera_0
WARNING: SAFETY VIOLATION: camera_0 - no_helmet for 5.2s (confidence: 0.87)
INFO: Incident video saved: incidents/violation_camera_0_20231119_153045.mp4
INFO: Notification sent successfully: camera_0_1700405445
```

### Níveis
- `DEBUG`: Tracking de violações frame-a-frame
- `INFO`: Início/fim de tracking, vídeos salvos, notificações
- `WARNING`: Violações detectadas
- `ERROR`: Falhas ao salvar vídeo ou enviar notificações

## 🔒 Segurança

### Boas Práticas
1. Use HTTPS para `BACKEND_API_URL` em produção
2. Implemente autenticação no backend
3. Limite permissões do diretório `incidents/`
4. Configure retention policy para vídeos antigos
5. Monitore uso de disco

### Exemplo de Limpeza de Vídeos Antigos

```bash
# Deletar vídeos com mais de 7 dias
find incidents/ -name "*.mp4" -mtime +7 -delete
```

## 🚀 Próximos Passos

- [ ] Adicionar suporte a webhooks
- [ ] Implementar retry automático para notificações
- [ ] Adicionar compressão de vídeo
- [ ] Dashboard web para visualização de alertas
- [ ] Integração com sistemas de alarme
