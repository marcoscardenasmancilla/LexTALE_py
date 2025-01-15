# LexTALE_PY
#
# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcoscardenasmancilla@gmail.com
# Date of creation          : 2024-11-27
# Licence                   : AGPL V3
# Copyright (c) 2024 Marcos H. Cárdenas Mancilla.
# 
# Descripción de LexTALE_PY:
# Este código Python analiza datos de tiempos de respuesta (RT) y precisión en una tarea de decisión léxica en inglés para determinar diferencias entre 2 grupos muestrales.
# Características del script:
# 1. preprocesa los datos, limpiando y transformando los RT y la proporción de respuestas correctas de palabras válidas y no válidas.
![% correct responses per word](https://github.com/user-attachments/assets/e6b310b2-9be1-4412-a441-b1d4df1a8bcc)
# 2. aplica una transformación Box-Cox a los RT para normalizar los datos.
![Box-Cox](https://github.com/user-attachments/assets/097746ab-b150-4e81-9c7d-89e267cfe22d)
# 3. realiza pruebas estadísticas, incluyendo Shapiro-Wilk, Levene, Mann-Whitney U y Kruskal-Wallis, para comparar las diferencias en RT entre grupos y también entre respuestas a palabras válidas y no válidas.
![RT words](https://github.com/user-attachments/assets/79d3847a-160b-47c3-8840-3fb47f4c629e)
# 4. realiza contrastes post-hoc (Dunn's Test y Mann-Whitney U).
# 5. calcula los tamaños del efecto para evaluar la significancia y magnitud de las diferencias encontradas.
![imagen](https://github.com/user-attachments/assets/8c7401e3-0b95-4315-8373-6523a145c295)
