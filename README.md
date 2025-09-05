# Curvas de Desembolso API

## Ajuste de curvas (`POST /api/curves/fit`)

Este endpoint ajusta curvas logísticas a series de desembolsos. Se debe enviar un `POST` a `/api/curves/fit` con un cuerpo JSON que incluya los filtros (`macrosectors`, `modalities`, etc.).

### `bandCoverage`

`bandCoverage` es opcional y se envía dentro del cuerpo JSON. Controla la cobertura de las bandas de cuantiles. Solo se permiten los valores `0.8`, `0.9` o `0.95`.

Ejemplo:

```http
POST /api/curves/fit
Content-Type: application/json

{
  "macrosectors": [11,22,33,44,55,66],
  "modalities": [111,222,333,444],
  "bandCoverage": 0.9
}
```

### Respuesta

La respuesta incluye la curva ajustada y las bandas estadísticas:

```json
{
  "bands": [
    {"k": 0, "hd": 0.0, "hd_up": 0.05, "hd_dn": 0.0}
  ],
  "bandsQuantile": [
    {"k": 0, "hd": 0.0, "hd_up": 0.06, "hd_dn": 0.0}
  ],
  "params": {
    "band_z": 1.2815515655446004
  }
}
```

- `bands`: bandas de amplitud fija calculadas usando `params.band_z`.
- `bandsQuantile`: bandas basadas en cuantiles, retornadas solo cuando se especifica `bandCoverage`.
- `params.band_z`: valor z usado para construir `bands`.

