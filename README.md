# ML_2026Q4 – Actividades evaluativas

## Integrantes

- Mayerly Andrea Suarez León
- Esteban Urrutia Bermudez
- Maycol Alejandro Vasquez Casas

## Datasets

| Actividad | Identificador | Enlace |
|---|---|---|
| EA1 y EA2 | `ADAMlam-16/coffee-data_eda-project` | https://huggingface.co/datasets/ADAMlam-16/coffee-data_eda-project |
| EA3 | `naabiil/Obesity_Levels_Estimation` | https://huggingface.co/datasets/naabiil/Obesity_Levels_Estimation |

- **Café (EA1 y EA2):** split `train` (10.000 filas, 16 columnas). Consumo de café y salud de personas de distintos países: edad, género, país, tazas de café al día, cafeína, horas y calidad del sueño, BMI, ritmo cardíaco, nivel de estrés, actividad física, problemas de salud, ocupación, tabaco y alcohol.
- **Obesidad (EA3):** splits `train` (1.688 filas) y `test` (423 filas), unidos en un DataFrame de 2.111 filas y 17 columnas. Estudio de la UCI sobre niveles de obesidad según hábitos alimenticios y condición física en personas de Colombia, Perú y México.

Los dos datasets son públicos y se descargan automáticamente desde Hugging Face con la clase `HFDatasetLoader`; no requieren archivos manuales ni token.

## Estructura del proyecto

```
.github/workflows/ea2.yml            Workflow de GitHub Actions (EA1, EA2 y EA3)
src/clases_po/
  hf_loader.py                       HFDatasetLoader: descarga el dataset y devuelve un DataFrame
  reportes.py                        Reportes: resume un DataFrame en un archivo .txt
  regresion_lineal.py                RegresionLineal: material de la clase de la semana 2
  clasificador_text.py               ClasificadorTexto: material de la clase de la semana 3
  clasificador_tabular.py            ClasificadorTabular: clasificador para datos tabulares (EA3)
src/Actividades_Evaluativas/
  EA1_carga_datos.ipynb              EA1: carga, transformaciones y resumen
  EA2_modelos.ipynb                  EA2: preprocesamiento, regresión y clasificación
  EA3_modelo.ipynb                   EA3: pregunta de negocio, Random Forest y resumen
  EA3_hallazgos.md                   EA3: documento de hallazgos y recomendaciones
  datos/                             Copias de los datasets generadas al ejecutar
  reportes/                          Resúmenes .txt generados al ejecutar
  graficas/                          Gráficas de la EA3 generadas al ejecutar
src/sincronicas/                     Prácticas de las clases sincrónicas
requirements.txt                     Dependencias
```

Las clases de `src/clases_po/` se construyeron siguiendo los ejemplos vistos en clase con el docente. `ClasificadorTabular` sigue la misma lógica de `ClasificadorTexto`: cada línea del código mínimo se convierte en un método (`preparar_datos`, `dividir_datos`, `construir_modelo`, `entrenar`, `evaluar`, `predecir` y `ejecutar`).

## Cómo ejecutar el proyecto

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# EA1
jupyter nbconvert --to notebook --execute --inplace src/Actividades_Evaluativas/EA1_carga_datos.ipynb

# EA2
jupyter nbconvert --to notebook --execute --inplace src/Actividades_Evaluativas/EA2_modelos.ipynb

# EA3
jupyter nbconvert --to notebook --execute --inplace src/Actividades_Evaluativas/EA3_modelo.ipynb
```

El entorno virtual se activa con `source` (el prompt muestra `(.venv)`). También se pueden abrir los notebooks en VS Code o Jupyter, seleccionar el kernel `.venv` y usar "Ejecutar todo".

## Ejecución automática (GitHub Actions)

El workflow `.github/workflows/ea2.yml` se ejecuta en cada `push` a `main`:

1. Instala las dependencias.
2. Borra los datos, reportes, gráficas y modelos de ejecuciones anteriores.
3. Ejecuta completos los notebooks de la EA1, la EA2 y la EA3: descarga de los datasets, preprocesamiento, entrenamiento y evaluación de los modelos, y generación de los resúmenes `.txt` y las gráficas.
4. Publica los resúmenes, las gráficas y los notebooks ejecutados como artefacto descargable (`resumen-actividades`).
5. Guarda los resultados generados en el repositorio con un commit automático de `github-actions[bot]`. Los modelos entrenados (`.pkl`) no se suben.

Si GitHub no muestra algún notebook, recargue la página o descargue el artefacto de la ejecución.

---

## EA3 – Aprendizaje supervisado

### Pregunta de negocio

**¿Qué hábitos de alimentación y de actividad física permiten identificar a las personas con obesidad, para que una EPS priorice a quién incluir en sus programas de prevención y acompañamiento nutricional?**

- **Quién necesita la respuesta:** el área de promoción y prevención de una EPS.
- **Qué decisión apoya:** a quién contactar primero para los programas de prevención y en qué hábitos enfocar las campañas.
- **Por qué este dataset permite responderla:** cada fila es una persona con sus hábitos (comida rápida, verduras, número de comidas, agua, actividad física, pantallas, alcohol, transporte e historial familiar) y su nivel de obesidad. Eso permite relacionar los hábitos con la obesidad sin necesidad de pesar ni medir a la persona.

### Variables

- **Objetivo:** `obesidad` = 1 si el nivel es `Obesity_Type_I`, `Obesity_Type_II` u `Obesity_Type_III`; 0 en los demás casos (46,6 % con obesidad).
- **Predictoras (14):** edad, frecuencia de verduras, comidas principales, agua, actividad física, tiempo frente a pantallas, come entre comidas, alcohol, historial familiar, comida calórica, fuma, controla calorías, género y transporte.
- **Excluidas por fuga de información:** estatura, peso y nivel de obesidad. El nivel de obesidad del estudio se calculó con el índice de masa corporal (peso / estatura²); si el modelo recibiera esas columnas, calcularía la respuesta con la fórmula en lugar de aprenderla de los hábitos.

### Tipo de aprendizaje y modelo

**Supervisado (clasificación).** La respuesta es conocida en los datos: cada persona ya está marcada con o sin obesidad. El modelo aprende de esos ejemplos y luego clasifica personas nuevas.

**Algoritmo: Random Forest.** Siguiendo la lógica vista en clase, primero se probó un modelo sencillo (árbol de decisión), se revisaron sus resultados y luego se cambió por uno mejor. El Random Forest obtuvo las mejores métricas, sobreajusta menos que un árbol solo y permite ver qué variables pesan más en la decisión.

#### Cómo funciona un árbol de decisión

1. Busca la pregunta (una variable y un valor de corte) que mejor separa a las personas con y sin obesidad. Por ejemplo: *¿tiene historial familiar de sobrepeso?*
2. Divide los datos en dos grupos según la respuesta.
3. Repite la búsqueda dentro de cada grupo, hasta llegar a grupos casi puros o al límite de profundidad.
4. Para clasificar a una persona nueva, se recorre el árbol respondiendo las preguntas hasta llegar a una hoja, que indica la clase.

El problema de un solo árbol profundo es que **memoriza** el entrenamiento: acierta casi todo con los datos que ya vio, pero se equivoca más con personas nuevas.

#### Cómo funciona Random Forest

1. Toma una **muestra aleatoria con reemplazo** de los datos de entrenamiento, así que cada árbol ve datos un poco distintos.
2. Construye un árbol de decisión con esa muestra, pero en cada división solo puede elegir entre **un subconjunto aleatorio de variables**. Así los árboles no se parecen entre sí.
3. Repite los pasos 1 y 2 hasta tener `n_estimators` árboles.
4. Para una persona nueva, **cada árbol vota** "con obesidad" o "sin obesidad". La probabilidad es la proporción de votos, y la clase final depende del umbral (por defecto, gana la mayoría).

Los errores de un árbol se compensan con los aciertos de los demás.

#### Parámetros principales

| Parámetro | Qué controla | Valor elegido |
|---|---|---|
| `n_estimators` | Número de árboles del bosque | 300 |
| `max_depth` | Profundidad máxima de cada árbol; limitarla reduce el sobreajuste | 15 |
| `min_samples_leaf` | Mínimo de personas en cada hoja; evita reglas basadas en casos aislados | 1 |
| `random_state` | Semilla para obtener siempre el mismo resultado | 42 |
| Umbral | Probabilidad mínima para marcar "con obesidad" | 0,4 |

Los tres primeros se eligieron con **validación cruzada de 5 particiones** sobre el conjunto de entrenamiento, comparando 16 combinaciones (F1 de validación del mejor: 0,912). El umbral se eligió con la regla *mayor recall con precisión de al menos 0,85*, porque para la EPS es más costoso dejar por fuera a una persona con obesidad que invitar a alguien que no lo necesitaba.

#### Ventajas y limitaciones

**Ventajas**
- Funciona bien con variables numéricas y categóricas, sin necesidad de escalar.
- Sobreajusta menos que un solo árbol, porque promedia muchos árboles distintos.
- Indica qué variables pesan más en la decisión.
- Entrega probabilidades, lo que permite ajustar el umbral según el costo de cada error.

**Limitaciones**
- Es menos interpretable que un solo árbol: no se pueden leer 300 árboles como reglas. Por eso se muestra el árbol de decisión del primer experimento para explicar las reglas principales.
- Es más lento de entrenar que un árbol solo.
- La importancia de variables tiende a favorecer variables con muchos valores distintos, como la edad.
- Muestra asociaciones, no causas.

### Resultados (conjunto de prueba, 418 personas)

| Modelo | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Árbol de decisión | 0,876 | 0,891 | 0,836 | 0,862 |
| Random Forest ajustado (umbral 0,5) | 0,914 | 0,939 | 0,872 | 0,904 |
| **Random Forest ajustado (umbral 0,4)** | **0,912** | **0,899** | **0,913** | **0,906** |
| Regresión logística balanceada | 0,758 | 0,698 | 0,851 | 0,767 |

Con el modelo final, de cada 100 personas con obesidad se identifican 91, y de cada 100 personas que se recomienda invitar, 90 realmente tienen obesidad.

**Documento de hallazgos con la respuesta a la pregunta, limitaciones y recomendaciones:** [`src/Actividades_Evaluativas/EA3_hallazgos.md`](src/Actividades_Evaluativas/EA3_hallazgos.md)

---

## Ubicación de las evidencias

| Evidencia | Ubicación |
|---|---|
| EA1 – Clase de carga | `src/clases_po/hf_loader.py` |
| EA1 – Clase de resumen | `src/clases_po/reportes.py` |
| EA1 – Notebook con cinco transformaciones | `src/Actividades_Evaluativas/EA1_carga_datos.ipynb` |
| EA1 – Resumen del DataFrame transformado | `src/Actividades_Evaluativas/reportes/EA1_resumen_transformado.txt` |
| EA2 – Construcción del DataFrame y preprocesamiento | `EA2_modelos.ipynb`, sección de carga y sección 1 |
| EA2 – Estado inicial de los datos | `src/Actividades_Evaluativas/reportes/reporte_dataframe_inicial.txt` |
| EA2 – Regresión lineal e interpretación | `EA2_modelos.ipynb`, sección 2 |
| EA2 – Clasificación e interpretación | `EA2_modelos.ipynb`, sección 3 |
| EA2 – Resumen automático de ambos modelos | `src/Actividades_Evaluativas/reportes/resumen_modelos.txt` |
| EA3 – Pregunta de negocio y construcción del DataFrame | `EA3_modelo.ipynb`, sección 1 |
| EA3 – Estado inicial de los datos | `src/Actividades_Evaluativas/reportes/EA3_reporte_dataframe_inicial.txt` |
| EA3 – Preprocesamiento (antes y después) | `EA3_modelo.ipynb`, sección 2 |
| EA3 – Clase del modelo | `src/clases_po/clasificador_tabular.py` |
| EA3 – Selección del modelo y ajuste de parámetros | `EA3_modelo.ipynb`, sección 3 |
| EA3 – Métricas, matriz de confusión, importancia y umbral | `EA3_modelo.ipynb`, sección 4 |
| EA3 – Gráficas | `src/Actividades_Evaluativas/graficas/` |
| EA3 – Resumen automático | `src/Actividades_Evaluativas/reportes/EA3_resumen_modelo.txt` |
| EA3 – Documento de hallazgos | `src/Actividades_Evaluativas/EA3_hallazgos.md` |
| Workflow (EA1, EA2 y EA3) | `.github/workflows/ea2.yml` |
