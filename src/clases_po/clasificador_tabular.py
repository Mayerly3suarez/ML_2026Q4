"""Flujo reutilizable para entrenar y evaluar un clasificador con datos tabulares.

Sigue la misma lógica de ``ClasificadorTexto`` vista en clase: cada línea del
código mínimo se convierte en un método:

    load_dataset   -> cargar_datos
    make_pipeline  -> construir_modelo
    fit            -> entrenar
    score          -> evaluar
    predict        -> predecir

La diferencia es el paso de transformación: en lugar de TF-IDF (texto ->
números), se usa un ``ColumnTransformer`` que codifica las columnas
categóricas con one-hot y, si se pide, escala las numéricas.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class ClasificadorTabular:
    """Clasifica filas de un DataFrame (por ejemplo, con o sin obesidad).

    Args:
        modelo: clasificador de scikit-learn sin entrenar. Por defecto,
            ``DecisionTreeClassifier``. Se copia con ``clone`` para que el
            mismo objeto pueda reutilizarse en varios experimentos.
        columnas_numericas: columnas numéricas (o ya codificadas como números).
        columnas_categoricas: columnas de texto que se codifican con one-hot.
        columna_objetivo: columna con la etiqueta (y).
        escalar: ``True`` aplica ``StandardScaler`` a las numéricas. Necesario
            para modelos basados en distancias (KNN, SVM) o en coeficientes
            (regresión logística); los árboles no lo necesitan.
        test_size: proporción reservada para prueba, entre 0 y 1.
        random_state: semilla para repetir la misma partición.
        directorio_modelos: carpeta donde se guardan los ``.pkl``.
    """

    def __init__(
        self,
        modelo: Any | None = None,
        columnas_numericas: list[str] | None = None,
        columnas_categoricas: list[str] | None = None,
        columna_objetivo: str = "objetivo",
        escalar: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        directorio_modelos: str | Path = "modelos",
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1, sin incluirlos.")
        self.columnas_numericas = list(columnas_numericas or [])
        self.columnas_categoricas = list(columnas_categoricas or [])
        if not self.columnas_numericas and not self.columnas_categoricas:
            raise ValueError("Indique al menos una columna numérica o categórica.")
        if columna_objetivo in self.columnas_numericas + self.columnas_categoricas:
            raise ValueError("La columna objetivo no puede estar entre las predictoras.")
        self.modelo = modelo if modelo is not None else DecisionTreeClassifier(random_state=random_state)
        self.columna_objetivo = columna_objetivo
        self.escalar = escalar
        self.test_size = test_size
        self.random_state = random_state
        self.directorio_modelos = Path(directorio_modelos)

        self.pipeline: Pipeline | None = None
        self.ruta_modelo: Path | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones: np.ndarray | None = None
        self.metricas: dict[str, Any] | None = None

    @property
    def predictoras(self) -> list[str]:
        return self.columnas_numericas + self.columnas_categoricas

    # ------------------------------------------------------------------ #
    # 1. cargar
    # ------------------------------------------------------------------ #
    @staticmethod
    def cargar_datos(fuente: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame desde un DataFrame, un .csv o Hugging Face.

        - ``pd.DataFrame``: se devuelve una copia.
        - ``"archivo.csv"``: se lee del disco.
        - Cualquier otro texto se trata como nombre de un dataset de
          Hugging Face; se unen todas sus particiones.
        """
        if isinstance(fuente, pd.DataFrame):
            return fuente.copy()
        ruta = str(fuente)
        if ruta.lower().endswith(".csv"):
            return pd.read_csv(ruta)
        from datasets import load_dataset  # import local: solo si se usa

        ds = load_dataset(ruta)
        return pd.concat([ds[p].to_pandas() for p in ds], ignore_index=True)

    # ------------------------------------------------------------------ #
    # preparar y dividir (lo que agrega la clase)
    # ------------------------------------------------------------------ #
    def preparar_datos(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Valida el DataFrame y devuelve X (predictoras) e y (etiqueta).

        - Verifica que existan todas las columnas.
        - Elimina filas con valores nulos en las predictoras o en el objetivo.
        - Exige al menos 2 clases y 2 ejemplos por clase (para estratificar).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        faltantes = set(self.predictoras + [self.columna_objetivo]) - set(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")

        datos = df[self.predictoras + [self.columna_objetivo]].copy()
        filas_iniciales = len(datos)
        datos = datos.dropna()

        conteo = datos[self.columna_objetivo].value_counts()
        if len(conteo) < 2:
            raise ValueError("Se necesitan al menos dos clases distintas.")
        if (conteo < 2).any():
            raise ValueError(f"Cada clase necesita al menos 2 ejemplos: {conteo.to_dict()}.")

        X = datos[self.predictoras].reset_index(drop=True)
        y = datos[self.columna_objetivo].reset_index(drop=True)
        print(
            f"[preparar_datos] {len(X)} filas válidas; "
            f"{filas_iniciales - len(X)} filas eliminadas; clases: {conteo.to_dict()}."
        )
        return X, y

    def dividir_datos(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en entrenamiento y prueba conservando la proporción de cada clase."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(
            f"[dividir_datos] stratify=y; entrenamiento={len(X_train)}; prueba={len(X_test)}; "
            f"random_state={self.random_state}."
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # 2. construir
    # ------------------------------------------------------------------ #
    def construir_modelo(self) -> Pipeline:
        """Encadena la transformación de columnas y el clasificador."""
        transformadores = []
        if self.columnas_numericas:
            numericas = StandardScaler() if self.escalar else "passthrough"
            transformadores.append(("num", numericas, self.columnas_numericas))
        if self.columnas_categoricas:
            transformadores.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.columnas_categoricas)
            )
        self.pipeline = Pipeline(
            [
                ("prep", ColumnTransformer(transformadores, verbose_feature_names_out=False)),
                ("modelo", clone(self.modelo)),
            ]
        )
        escala = "StandardScaler" if self.escalar else "sin escalar"
        print(f"[construir_modelo] numéricas ({escala}) + one-hot -> {type(self.modelo).__name__}")
        return self.pipeline

    # ------------------------------------------------------------------ #
    # 3. entrenar
    # ------------------------------------------------------------------ #
    def entrenar(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
        """Ajusta el pipeline y lo guarda como .pkl con fecha y hora.

        La codificación (y el escalado, si se usa) aprende solo con X_train
        (fit); a los datos de prueba solo los transforma al predecir: no se
        espía el examen.
        """
        if self.pipeline is None:
            self.construir_modelo()
        self.pipeline.fit(X_train, y_train)

        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nombre = type(self.modelo).__name__
        self.ruta_modelo = self.directorio_modelos / f"{fecha}_{nombre}.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.pipeline, archivo)
        print(f"[entrenar] {len(X_train)} filas; modelo guardado en {self.ruta_modelo}")
        return {"pipeline": self.pipeline, "ruta_modelo": self.ruta_modelo, "fecha_modelo": fecha}

    def cargar_modelo(self, ruta_modelo: str | Path) -> Pipeline:
        """Carga un .pkl guardado por ``entrenar`` para volver a usarlo."""
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")
        with ruta.open("rb") as archivo:
            self.pipeline = pickle.load(archivo)
        self.ruta_modelo = ruta
        return self.pipeline

    # ------------------------------------------------------------------ #
    # 5. predecir
    # ------------------------------------------------------------------ #
    def _modelo_listo(self) -> Pipeline:
        if self.pipeline is None or not hasattr(self.pipeline, "classes_"):
            raise RuntimeError("Primero entrena el modelo (entrenar o ejecutar) o cárgalo (cargar_modelo).")
        return self.pipeline

    def predecir_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Probabilidad de cada clase por fila (una columna por clase)."""
        pipe = self._modelo_listo()
        if not hasattr(pipe, "predict_proba"):
            raise TypeError(f"{type(self.modelo).__name__} no entrega probabilidades.")
        return pd.DataFrame(pipe.predict_proba(X[self.predictoras]), columns=pipe.classes_)

    def predecir(
        self,
        X: pd.DataFrame,
        umbral: float | None = None,
        clase_umbral: Any = 1,
    ) -> np.ndarray:
        """Devuelve la clase de cada fila.

        Sin ``umbral``: gana la clase más probable.
        Con ``umbral``: se predice ``clase_umbral`` si su probabilidad es
        ``>= umbral``; si no, gana la más probable entre las demás clases.
        """
        pipe = self._modelo_listo()
        if umbral is None:
            return pipe.predict(X[self.predictoras])
        if not 0 <= umbral <= 1:
            raise ValueError("umbral debe estar entre 0 y 1.")
        proba = self.predecir_proba(X)
        if clase_umbral not in proba.columns:
            raise ValueError(f"'{clase_umbral}' no es una clase del modelo: {list(proba.columns)}.")
        otras = proba.drop(columns=[clase_umbral])
        return np.where(proba[clase_umbral] >= umbral, clase_umbral, otras.idxmax(axis=1))

    # ------------------------------------------------------------------ #
    # 4. evaluar
    # ------------------------------------------------------------------ #
    def evaluar(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        umbral: float | None = None,
        clase_umbral: Any = 1,
    ) -> dict[str, Any]:
        """Calcula exactitud, precisión, recall y F1 por clase, macro y ponderado, y la matriz de confusión."""
        pred = self.predecir(X_test, umbral=umbral, clase_umbral=clase_umbral)
        self.predicciones = pred
        clases = list(self._modelo_listo().classes_)
        p, r, f, n = precision_recall_fscore_support(y_test, pred, labels=clases, zero_division=0)
        por_clase = pd.DataFrame({"precision": p, "recall": r, "f1": f, "casos": n}, index=clases)
        matriz = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=clases),
            index=[f"real {c}" for c in clases],
            columns=[f"pred {c}" for c in clases],
        )
        self.metricas = {
            "exactitud": float(accuracy_score(y_test, pred)),
            "f1_macro": float(por_clase["f1"].mean()),
            "f1_ponderado": float(np.average(por_clase["f1"], weights=por_clase["casos"])),
            "por_clase": por_clase,
            "matriz": matriz,
            "umbral": umbral,
        }
        print(f"[evaluar] exactitud={self.metricas['exactitud']:.4f}; "
              f"F1 macro={self.metricas['f1_macro']:.4f}; F1 ponderado={self.metricas['f1_ponderado']:.4f}")
        print(classification_report(y_test, pred, labels=clases, zero_division=0, digits=4))
        print(matriz, "\n")
        return self.metricas

    # ------------------------------------------------------------------ #
    # 6. interpretar
    # ------------------------------------------------------------------ #
    def importancia_variables(self, n: int = 15) -> pd.Series:
        """Variables que más pesan en la decisión del modelo.

        Árboles y Random Forest: ``feature_importances_``.
        Modelos lineales (regresión logística): valor absoluto de los coeficientes.
        """
        pipe = self._modelo_listo()
        modelo = pipe.named_steps["modelo"]
        nombres = pipe.named_steps["prep"].get_feature_names_out()
        if hasattr(modelo, "feature_importances_"):
            valores = modelo.feature_importances_
        elif hasattr(modelo, "coef_"):
            valores = np.abs(modelo.coef_).mean(axis=0)
        else:
            raise TypeError(f"{type(modelo).__name__} no entrega importancia de variables.")
        return pd.Series(valores, index=nombres).sort_values(ascending=False).head(n)

    # ------------------------------------------------------------------ #
    # todo el flujo
    # ------------------------------------------------------------------ #
    def ejecutar(self, fuente: str | Path | pd.DataFrame) -> dict[str, Any]:
        """Ejecuta en orden: cargar, preparar, dividir, construir, entrenar y evaluar."""
        print(f"[ejecutar][1/6] cargar_datos ({type(self.modelo).__name__})")
        df = self.cargar_datos(fuente)
        print("[ejecutar][2/6] preparar_datos")
        X, y = self.preparar_datos(df)
        print("[ejecutar][3/6] dividir_datos")
        self.X_train, self.X_test, self.y_train, self.y_test = self.dividir_datos(X, y)
        print("[ejecutar][4/6] construir_modelo")
        self.construir_modelo()
        print("[ejecutar][5/6] entrenar")
        entrenamiento = self.entrenar(self.X_train, self.y_train)
        print("[ejecutar][6/6] evaluar")
        metricas = self.evaluar(self.X_test, self.y_test)
        print("[ejecutar] flujo completo terminado.")
        return {
            "ruta_modelo": entrenamiento["ruta_modelo"],
            "metricas": metricas,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "predicciones": self.predicciones,
        }
