# Guía de retoque — variables y uso

Guía completa en español de todas las variables de retoque del pipeline: qué hacen,
qué valores usar y cómo aplicarlas. (Documentación general del proyecto en inglés en
[README.md](README.md).)

## Dos formas de aplicar el retoque

1. **Automática**: la IA analiza cada foto (Etapa 3) y rellena todas estas variables sola.
   Solo tienes que dejar correr el pipeline con `python -m pipeline.run`.
2. **Manual**: tú decides los valores por foto con el comando `redo`, que vuelve a revelar y
   retocar sin repetir el análisis:

```bash
venv/bin/python -m pipeline.redo --list      # ver las fotos de la cola con su id
venv/bin/python -m pipeline.redo 6 --show     # ver los valores actuales de la foto 6
```

Casi todas las variables van de **0.0 a 1.0** (0 = desactivada, valores más altos = efecto
más fuerte). La filosofía es "suave y natural": rara vez conviene pasar de ~0.4.

## Variables de cara

| Variable | Flag CLI | Rango | Qué hace | Valor recomendado |
|---|---|---|---|---|
| `skin_smoothing` | `--skin` | 0–1 | Suaviza la piel con separación de frecuencias: iguala el tono pero **conserva los poros** | 0.2–0.4 |
| `remove_blemishes` | `--blemishes` / `--no-blemishes` | sí/no | Borra granos, marcas de acné, verrugas y pelos sueltos clonando piel vecina. **Nunca** toca lunares ni rasgos permanentes | actívalo si hay imperfecciones temporales |
| `skin_tone_correction` | `--skintone` | 0–1 | Empareja el tono desigual/manchado y da un color saludable (mantiene la textura) | 0.2–0.4 |
| `reduce_dark_circles` | `--darkcircles` | 0–1 | Aclara ojeras y bolsas bajo los ojos hacia el brillo de la piel de alrededor | 0.2–0.4 |
| `reduce_dewlap` | `--dewlap` | 0–1 | Reduce la **papada** / doble mentón: eleva suavemente la piel bajo la barbilla (deformación sutil, controlada) | 0.2–0.4 |
| `brighten_eyes` | `--eyes` | 0–1 | Ilumina **el blanco del ojo** (esclerótica) y reduce el enrojecimiento | 0.1–0.3 |
| `iris_enhance` | `--iris` | 0–1 | Da un toque sutil de saturación/nitidez **solo al iris** (ni pupila ni blanco) | 0.2–0.4 |
| `whiten_teeth` | `--teeth` | 0–1 | Desatura el amarillo y aclara los dientes | 0.1–0.3 |

## Variable de ropa

| Variable | Flag CLI | Rango | Qué hace | Recomendado |
|---|---|---|---|---|
| `clothing_contrast` | `--clothing` | 0–1 | Realza el contraste/textura local de la ropa y telas (excluye la piel) | 0.2–0.4 |

## Fondo (`background`)

Se controla con `--background` y tiene 5 modos:

| Modo | Qué hace |
|---|---|
| `keep` | Deja el fondo igual (por defecto, casi siempre lo correcto) |
| `blur` | Desenfoque tipo bokeh, para fondos ocupados o que distraen |
| `smooth` | **Quita arrugas** de un fondo de estudio/papel/tela: clona las imperfecciones y suaviza los pliegues |
| `studio` | Reemplaza el fondo por un gris de estudio neutro |
| `replace` | Genera un fondo nuevo con IA (necesita ComfyUI). Descríbelo con `--prompt`. Si ComfyUI no está, cae automáticamente a `studio` |

## Balance de blancos (etapa de revelado)

El balance de blancos se ajusta en el **revelado**, no en el retoque, y por defecto usa el
balance **de la cámara** (tal como se tomó la foto), que es el más preciso. Puedes cambiarlo:

| Modo | Flag CLI | Qué hace |
|---|---|---|
| `camera` | `--wb camera` | Usa el balance original de la cámara (preciso, **por defecto**) |
| `kelvin` | `--wb 5200` | Fija una temperatura de color en Kelvin. **Más alto = más cálido**; 6500 = sin cambio |

- `--tint N` → ajuste verde-magenta (−50 verde .. +50 magenta), solo en modo Kelvin.
- Cambiar el balance de blancos **vuelve a revelar** la foto desde el RAW automáticamente.

```bash
venv/bin/python -m pipeline.redo 6 --wb camera          # balance original (preciso)
venv/bin/python -m pipeline.redo 6 --wb 5200            # más frío
venv/bin/python -m pipeline.redo 6 --wb 7000 --tint 5   # más cálido, leve magenta
```

## Intensidad general (`intensity`)

Multiplica **todos** los efectos a la vez con `--intensity`:

- `subtle` → ×0.6 (apenas perceptible)
- `natural` → ×1.0 (por defecto)
- `polished` → ×1.3 (estilo editorial)

## Otros flags útiles del comando `redo`

- `--prompt "texto"` → descripción del fondo cuando usas `--background replace`
- `--from-raw` → vuelve a revelar desde el RAW original en vez de reusar el TIFF
- `--list` → lista las fotos de la cola
- `--show` → muestra los valores actuales sin renderizar

## Ejemplos completos

```bash
# Retrato con retoque suave y natural de piel y ojos
venv/bin/python -m pipeline.redo 6 --skin 0.3 --eyes 0.25 --iris 0.3

# Retoque completo de cara: piel, imperfecciones, tono, ojeras, papada, ojos e iris
venv/bin/python -m pipeline.redo 6 --skin 0.3 --blemishes --skintone 0.3 \
    --darkcircles 0.3 --dewlap 0.3 --eyes 0.25 --iris 0.3 --teeth 0.2

# Piel + más definición en la ropa, con intensidad editorial
venv/bin/python -m pipeline.redo 6 --skin 0.35 --clothing 0.4 --intensity polished

# Desenfocar un fondo desordenado
venv/bin/python -m pipeline.redo 6 --background blur

# Quitar arrugas de un fondo de estudio
venv/bin/python -m pipeline.redo 6 --background smooth

# Reemplazar el fondo generándolo con IA
venv/bin/python -m pipeline.redo 6 --background replace --prompt "luz suave de ventana, interior claro"
```

## Notas importantes

- Los valores que pongas se **guardan** en la base de datos, así que se mantienen si vuelves a
  renderizar la foto más tarde.
- Cualquier flag de cara activa automáticamente `is_portrait=true` (la variable interna que
  indica que hay un rostro; no necesitas ponerla a mano).
- El resultado se publica en `data/output/` como **JPEG al 90 %** (unos pocos MB); los TIFF de
  16 bits intermedios se borran solos (usa `PIPELINE_KEEP_TIFFS=1` para conservarlos).
- La **exposición y el balance de blancos global** se ajustan antes, en la etapa de revelado
  (no en retoque); `skin_tone_correction` es la parte de corrección de color *local* de la piel.
- Todo el retoque es **procesamiento determinista de imagen** (OpenCV), no IA generativa: nunca
  regenera la foto, así que la identidad, la textura y la resolución siempre se conservan. La
  única excepción es `background replace`, que sí usa IA solo para el fondo.
