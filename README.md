# LexTALE_py
#
# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcos.cardenas.m@usach.cl
# Date of creation          : 2024-11-27
# Licence                   : AGPL V3
# Copyright (c) 2024 Marcos H. Cárdenas Mancilla.
# 
# Descripción de LexTALE_PY:
# Este código Python analiza datos de tiempos de respuesta (RT) y precisión en una tarea de decisión léxica en inglés, diferenciando entre participantes con y sin experiencia en traducción.
# Características del script:
# 1. preprocesa los datos, limpiando y transformando los RT y la proporción de respuestas correctas de palabras válidas y no válidas.
# 2. aplica una transformación Box-Cox a los RT para normalizar los datos.
# 3. realiza pruebas estadísticas, incluyendo Shapiro-Wilk, Levene, Mann-Whitney U y Kruskal-Wallis, para comparar las diferencias en RT entre grupos y también entre respuestas a palabras válidas y no válidas.
# 4. realiza contrastes post-hoc (Dunn's Test y Mann-Whitney U).
# 5. calcula los tamaños del efecto para evaluar la significancia y magnitud de las diferencias encontradas. 
# El objetivo del análisis es determinar si la experiencia en traducción afecta significativamente los tiempos de respuesta y la precisión en la tarea de decisión léxica.
