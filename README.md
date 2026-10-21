# ML_2026Q4 – Actividades evaluativas

## Integrantes

- Mayerly Andrea Suarez León
- Esteban Urrutia Bermudez
- Maycol Alejandro Vasquez Casas

## Dataset

- **Identificador:** `ADAMlam-16/coffee-data_eda-project`
- **Enlace:** https://huggingface.co/datasets/ADAMlam-16/coffee-data_eda-project
- **Split:** `train` (10.000 filas, 16 columnas)
- **Contenido:** consumo de café y salud de personas de distintos países: edad, género, país, tazas de café al día, cafeína, horas y calidad del sueño, BMI, ritmo cardíaco, nivel de estrés, actividad física, problemas de salud, ocupación, tabaco y alcohol.

El dataset es público y se descarga automáticamente desde Hugging Face con la clase `HFDatasetLoader`; no requiere archivos manuales ni token.

## Estructura del proyecto

```
.github/workflows/ea2.yml            Workflow de GitHub Actions
src/clases_po/
  hf_loader.py                       HFDatasetLoader: descarga el dataset y devuelve un DataFrame
  reportes.py                        Reportes: resume un DataFrame en un archivo .txt
  regresion_lineal.py                RegresionLineal: material de la clase de la semana 2
src/Actividades_Evaluativas/
  EA1_carga_datos.ipynb              EA1: carga, transformaciones y resumen
  EA2_modelos.ipynb                  EA2: preprocesamiento, regresión y clasificación
  datos/                             Copia del dataset generada al ejecutar
  reportes/                          Resúmenes .txt generados al ejecutar
src/sincronicas/                     Prácticas de las clases sincrónicas
requirements.txt                     Dependencias
```

Las clases de `src/clases_po/` se construyeron siguiendo los ejemplos vistos en clase con el docente.

## Cómo ejecutar el proyecto

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# EA1
jupyter nbconvert --to notebook --execute --inplace src/Actividades_Evaluativas/EA1_carga_datos.ipynb

# EA2
jupyter nbconvert --to notebook --execute --inplace src/Actividades_Evaluativas/EA2_modelos.ipynb
```

El entorno virtual se activa con `source` (el prompt muestra `(.venv)`). También se pueden abrir los notebooks en VS Code o Jupyter, seleccionar el kernel `.venv` y usar "Ejecutar todo".

## Ejecución automática (GitHub Actions)

El workflow `.github/workflows/ea2.yml` se ejecuta en cada `push` a `main`: instala las dependencias, borra los resultados anteriores, ejecuta los notebooks de la EA1 y la EA2 completos (descarga del dataset, transformaciones, preprocesamiento, entrenamiento y evaluación de ambos modelos), publica los resúmenes `.txt` y los notebooks ejecutados como artefacto descargable (`resumen-modelos`) y guarda los resultados generados en el repositorio con un commit automático de `github-actions[bot]`.

- **Ejecución exitosa:** (https://github.com/Mayerly3suarez/ML_2026Q4/actions/runs/39536285384)

Si GitHub no muestra algún notebook, recargue la página o descargue el artefacto de la ejecución.

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
| EA2 – Workflow | `.github/workflows/ea2.yml` |
