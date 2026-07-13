# Guía de retoque — variables y uso

Guía completa en español de todas las variables de retoque del pipeline: qué hacen,
qué valores usar y cómo aplicarlas. (Documentación general del proyecto en inglés en
[README.md](README.md).)

## Dos formas de aplicar el retoque

1. **Automática**: la IA analiza cada foto (Etapa 3) y rellena todas estas variables sola.
   Solo tienes que dejar correr el pipeline con `python -m pipeline.run`.
   - Puedes **forzar valores en todas las fotos** pasando los mismos flags al pipeline; se
     aplican por encima de lo que decida la IA:
     ```bash
     python -m pipeline.run --skin 0.3 --dewlap 0.3 --wb camera --background blur
     ```
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
| `skin_smoothing` | `--skin` | 0–1 | Suaviza la piel con separación de frecuencias: iguala el tono y **atenúa la textura fina (poros/ruido) según la fuerza** (≈−13% a 0.2, −33% a 0.5, −58% a 0.9), conservando bordes/rasgos y algo de poro para que no se vea plástico | 0.3–0.5 |
| `remove_blemishes` | `--blemishes` / `--no-blemishes` | sí/no | Borra granos, marcas de acné, verrugas y pelos sueltos clonando piel vecina. **Nunca** toca lunares ni rasgos permanentes | actívalo si hay imperfecciones temporales |
| `skin_tone_correction` | `--skintone` | 0–1 | Empareja el tono desigual/manchado y lleva la piel al **color saludable propio de su tipo de piel** (mantiene textura y variación). Ya no aplica un empujón cálido fijo a todos por igual | 0.2–0.4 |
| `skin_type` | `--skin-type` | auto/fair/light/medium/olive/tan/brown/deep | Tipo de piel del sujeto para que la corrección apunte al **tono (hue) correcto**: rota el color de la piel hacia un tono cálido anaranjado (~48-50°) según el tipo, así la piel amarillenta/cetrina gana naranja. Ahora cubre también la **piel del cuerpo** (brazos, hombros, escote), no solo la cara, para que no queden con un tono más amarillo. La IA lo detecta o se mide por **ITA°**; `--skin-type` lo fuerza | auto |
| `skin_warmth` | `--skin-warmth` | −1…1 | Sesga el tono de piel hacia el **naranja** (+) o el amarillo (−). Súbelo si la piel se ve muy amarilla. Global con `PIPELINE_SKIN_WARMTH` | 0.25 |
| `reduce_dark_circles` | `--darkcircles` | 0–1 | Aclara ojeras y bolsas bajo los ojos hacia el brillo de la piel de alrededor | 0.2–0.4 |
| `reduce_dewlap` | `--dewlap` | 0–1 | Reduce la **papada** / doble mentón: eleva suavemente la piel bajo la barbilla (deformación sutil, controlada) | 0.2–0.4 |
| `brighten_eyes` | `--eyes` | 0–1 | Ilumina **el blanco del ojo** (esclerótica) y reduce el enrojecimiento | 0.1–0.3 |
| `iris_enhance` | `--iris` | 0–1 | Da un toque sutil de saturación/nitidez **solo al iris** (ni pupila ni blanco) | 0.2–0.4 |
| `whiten_teeth` | `--teeth` | 0–1 | Desatura el amarillo y aclara los dientes | 0.1–0.3 |
| `lip_enhance` | `--lips` | 0–1 | Recupera/intensifica el **rojo natural de los labios** + un poco de riqueza y definición (detecta el labio por color, no toca dientes ni brackets) | 0.3–0.4 |
| `hair_texture` | `--hair` / `--no-hair` | 0–1 | **Textura y claridad del cabello**: resalta los mechones. **Activado por defecto** en retratos con cabello visible | 0.4–0.6 |
| `hair_shimmer` | `--hair-shimmer` | 0–1 | Brillo del cabello: sube las luces y baja las sombras (opcional) | 0 |
| `hair_defrizz` | `--defrizz` | 0–1 | Reduce el frizz / pelos sueltos (cabello más liso; opcional) | 0 |

## Variable de ropa

| Variable | Flag CLI | Rango | Qué hace | Recomendado |
|---|---|---|---|---|
| `clothing_contrast` | `--clothing` | 0–1 | Realza el contraste/textura local de la ropa y telas (excluye la piel) | 0.2–0.4 |
| `clothing.color` | `--cloth-color` | nombre | Color dominante de la ropa que **identifica la IA** (negro/blanco/rojo/azul…). Decide el tratamiento: las telas neutras no se saturan | auto (IA) |
| `clothing.color_pop` | `--cloth-pop` | 0–1 | **Realza el color** de la tela (saturación ponderada por vibrance). 0 para telas negras/blancas/grises | 0.2–0.4 en color |
| `clothing.luminance` | `--cloth-luminance` | −1…1 | Aclara (+) u oscurece (−) la ropa | ±0.2 |
| `clothing.shadows` | `--cloth-shadows` | −1…1 | Sube (+) o profundiza (−) las sombras de la tela (más profundidad) | −0.2 |
| `clothing.blacks` | `--cloth-blacks` | −1…1 | Punto de negro de la ropa; negativo = negros ricos y profundos (ideal para prendas oscuras) | −0.2 a −0.3 |
| `clothing.whites` | `--cloth-whites` | −1…1 | Punto de blanco; negativo protege/recupera (ideal para prendas blancas, sin quemarse) | −0.1 a −0.2 |
| `subject_exposure` | `--subject-exposure` | EV (pasos) | **Sube (o baja) la exposición solo del sujeto**, sin tocar el fondo (usa la máscara del sujeto). Útil si el sujeto está a contraluz o subexpuesto. +0.5 = más claro, −0.3 = más oscuro | 0 |
| `auto_levels` | `--auto-levels` | on/off | **Exposición automática por capas.** Mide por separado la **piel de la cara**, el **resto del sujeto** (cuerpo/ropa/pelo) y el **fondo**, y lleva cada zona a su tono medio objetivo usando las máscaras que el retoque ya calcula. Las capas no se solapan, así que aclarar una cara oscura no aclara el fondo; si tras medir la piel aún queda quemada, se recupera sola. Corre sobre el TIFF ya revelado (sin revelar de nuevo) | off |
| `auto_skin_target` / `auto_subject_target` / `auto_background_target` | `--skin-target` / `--subject-target` / `--background-target` | 0–1 | Tono medio objetivo de cada capa (implican `--auto-levels`). Por defecto piel 0.72, sujeto 0.5, fondo apagado. La corrección se limita a ±1.25 pasos por capa para no exagerar | 0.72 / 0.5 / — |

## Fondo (`background`)

Por defecto **siempre `keep`**: el fondo nunca se cambia automáticamente. Solo se modifica si
pasas explícitamente `--background`. Modos:

| Modo | Qué hace |
|---|---|
| `auto` | **Decide solo** según el fondo: fondo de estudio neutro que llena el cuadro con **arrugas → `smooth`**; fondo que solo llena parte del cuadro con **otras zonas visibles (suelo, soportes, pared) → `studio`** con el tono del propio fondo (blanco/gris/negro según su luminosidad); **cualquier otra cosa → `keep`** |
| `keep` | Deja el fondo igual (**por defecto siempre**) |
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

# Corregir el tono al color correcto según el tipo de piel
venv/bin/python -m pipeline.redo 6 --skintone 0.4 --skin-type deep    # piel oscura: marrón cálido
venv/bin/python -m pipeline.redo 6 --skin-type olive                  # corrige el matiz verde

# Piel + más definición en la ropa, con intensidad editorial
venv/bin/python -m pipeline.redo 6 --skin 0.35 --clothing 0.4 --intensity polished

# Revelado de la ropa según su color (la IA lo detecta; aquí se fuerza)
venv/bin/python -m pipeline.redo 6 --cloth-color red --cloth-pop 0.35 --cloth-shadows -0.15   # realza un vestido rojo
venv/bin/python -m pipeline.redo 6 --cloth-color black --cloth-blacks -0.25                   # negros profundos
venv/bin/python -m pipeline.redo 6 --cloth-color white --cloth-whites -0.15                   # blancos limpios sin quemar

# Subir la exposición solo del sujeto (+0.6 EV), fondo intacto
venv/bin/python -m pipeline.redo 6 --subject-exposure 0.6

# Exposición automática por capas (piel/sujeto/fondo a su objetivo); arregla piel quemada
venv/bin/python -m pipeline.redo 99 --auto-levels
venv/bin/python -m pipeline.redo 99 --skin-target 0.7 --subject-target 0.5   # ajustar objetivos

# Volver a analizar la foto con la IA (nuevo análisis) y renderizar
venv/bin/python -m pipeline.redo 6 --reanalyze
venv/bin/python -m pipeline.redo 6 --reanalyze --skin 0.3   # reanaliza + ajustes manuales

# Ver la máscara de piel (original | selección en rojo) para inspeccionarla
venv/bin/python -m pipeline.redo 6 --dump-mask   # -> data/output/_mask_<nombre>_<id>.jpg

# Fondo automático: decide smooth/studio/keep según la escena
venv/bin/python -m pipeline.redo 6 --background auto

# Desenfocar un fondo desordenado
venv/bin/python -m pipeline.redo 6 --background blur

# Quitar arrugas de un fondo de estudio
venv/bin/python -m pipeline.redo 6 --background smooth

# Reemplazar el fondo generándolo con IA
venv/bin/python -m pipeline.redo 6 --background replace --prompt "luz suave de ventana, interior claro"

# Borrar objetos: elimina las zonas blancas de la máscara y rellena el hueco
venv/bin/python -m pipeline.redo 6 --erase-mask 6.png                    # relleno según contenido (rápido)
venv/bin/python -m pipeline.redo 6 --erase-method generative --erase-prompt "césped vacío"  # relleno generativo (ComfyUI)
venv/bin/python -m pipeline.redo 6 --clear-erase                         # olvidar la máscara guardada
```

## Interfaz web (revisar y rehacer en lote)

```bash
venv/bin/python -m pipeline.webui                  # http://127.0.0.1:8765 (solo local)
venv/bin/python -m pipeline.webui --host 0.0.0.0   # accesible desde la red local
```

Una galería web sobre la cola (sin dependencias nuevas): muestra cada foto con su último
render, filtros por estado y búsqueda. Al hacer clic se abre la foto a tamaño completo, con
comparación contra la vista previa original y los parámetros exactos usados. **Zoom** con los
botones +/−/Fit, la rueda del ratón, arrastrar para desplazar o doble clic; **Redo this…** abre
el asistente solo para esa foto.

- **Recortar** — arrastra un rectángulo (con relación de aspecto opcional: 1:1, 4:5, 5:7, 3:2,
  2:3, 16:9) sobre la imagen a fotograma completo y aplícalo; vuelve a revelar el RAW con el
  nuevo recorte.
- **Borrar objetos** — pinta con un pincel sobre lo que debe desaparecer (un cable, una persona,
  pelusa en el fondo), elige **relleno según contenido** (rápido, determinista) o **relleno
  generativo** (ComfyUI, con prompt opcional) y re-renderiza. La máscara queda guardada con la
  foto (en `data/masks/`) hasta que la borres desde la misma barra.
- **Asistente de rehacer en lote** — selecciona fotos (shift-clic para rangos) y *Redo
  selected…* abre un asistente por pasos (Piel → Cara → Pelo y ropa → Luz y revelado → Fondo →
  Revisión) con todos los parámetros como controles. Solo se aplican los parámetros que actives;
  el resto conserva los valores actuales de cada foto. En la revisión se ve el comando `redo`
  equivalente, más *from RAW*, *re-analizar con IA* y la medición del análisis (*solo sujeto*,
  *exposición de piel*).
- **Panel de trabajos** — los renders corren en cola, uno a uno, por el mismo código que
  `pipeline.redo`, con progreso por foto, errores y cancelación; la galería se refresca sola.

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
  regenera la foto, así que la identidad, la textura y la resolución siempre se conservan. Las
  únicas excepciones usan IA solo donde tú lo pides: `background replace` (el fondo) y el
  borrado de objetos con `--erase-method generative` (solo el área que pintaste).
