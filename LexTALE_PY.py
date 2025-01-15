# ==================================================================================================================================================
# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcoscardenasmancilla@gmail.com
# Date of creation          : 2024-11-27
# Licence                   : AGPL V3
# Copyright (c) 2024 Marcos H. Cárdenas Mancilla.
# ==================================================================================================================================================
# Descripción de LexTALE_PY:
# Este código Python analiza datos de tiempos de respuesta (RT) y precisión en una tarea de decisión léxica en inglés (Lemhöfer y Broersma, 2012), 
# para determinar diferencias entre grupos muestrales.
# Características del script:
# 1. preprocesa los datos, limpiando y transformando los RT y la proporción de respuestas correctas de palabras válidas y no válidas.
# 2. aplica una transformación Box-Cox a los RT para normalizar los datos.
# 3. realiza pruebas estadísticas e.g., Shapiro-Wilk, Levene, Mann-Whitney U y Kruskal-Wallis, para comparar las diferencias en RT entre grupos, 
# y también entre respuestas a palabras válidas y no válidas.
# 4. realiza contrastes post-hoc (Dunn's Test y Mann-Whitney U).
# 5. calcula los tamaños del efecto para evaluar la significancia y magnitud de las diferencias encontradas. 
# ==================================================================================================================================================

# Carga de librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu, probplot, boxcox, kruskal, levene
from itertools import combinations
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 0. Descripcion de metadatos
metadata_description = [
    "participant: Identificador del participante.",
    "experience: Experiencia (e.g., sin = Exp_0; con = Exp_1)",
    "text: Texto mostrado durante la tarea.",
    "response_time: Tiempo de respuesta en milisegundos.",
    "correct: Indica si la respuesta fue correcta ('yes') o incorrecta ('no')."
]
print("Descripcion de Metadatos:")
for item in metadata_description:
    print(f"- {item}")

# 1. Preprocesamiento de datos
# Cargar datos desde CSV
data_path = r'lextale_results_long_format.csv'
df = pd.read_csv(data_path)

# Verificar valores únicos en la columna 'correct'
print("Valores únicos en la columna 'correct':")
print(df['correct'].unique())

# Limpiar valores problemáticos en la columna 'correct'
df['correct'] = df['correct'].str.strip().str.lower()
df = df[df['correct'].isin(['yes', 'no'])]

# Convertir columna 'correct' a valores numéricos
df['correct_numeric'] = df['correct'].map({'yes': 1, 'no': 0})

# Estimar la proporción de respuestas correctas por palabra mostrada
proportion_correct_wd = df.groupby('word_shown')['correct_numeric'].mean().reset_index()
proportion_correct_wd.columns = ['Word_Shown', 'Proportion_Correct']
# Imprimir la tabla de salida para la proporción de respuestas correctas por palabra mostrada
print("Tabla de Proporción de Respuestas Correctas por Palabra:")
print(proportion_correct_wd)

proportion_correct_exp = df.groupby('experience')['correct_numeric'].mean().reset_index()
proportion_correct_exp.columns = ['Experience', 'Proportion_Correct']
print("Tabla de Proporción de Respuestas Correctas por Experiencia:")
print(proportion_correct_exp)

# Visualizar la proporción de respuestas correctas por Palabras
plt.figure(figsize=(10, 6))
plt.bar(proportion_correct_wd['Word_Shown'], proportion_correct_wd['Proportion_Correct'], color=sns.color_palette('colorblind'))
plt.xlabel('Word Shown')
plt.ylabel('Proportion Correct')
plt.title('Proportion of Correct Responses by Word')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Visualizar la proporción de respuestas correctas por Experiencia
plt.figure(figsize=(10, 6))
plt.bar(proportion_correct_exp['Experience'], proportion_correct_exp['Proportion_Correct'], color=sns.color_palette('colorblind'))
plt.xlabel('Experience')
plt.ylabel('Proportion Correct')
plt.title('Proportion of Correct Responses by Experience')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Filtrar solo respuestas correctas
df_correct = df[df['correct'] == 'yes']

# Verificar si el filtrado dejó suficientes datos
if df_correct.empty:
    print("Error: No hay datos después de filtrar por respuestas correctas ('yes'). Verifica los datos.")
else:
    # Verificar si la columna 'response_time' tiene valores negativos o no numéricos
    if df_correct['response_time'].dtype != np.number:
        df_correct['response_time'] = pd.to_numeric(df_correct['response_time'], errors='coerce')
    df_correct = df_correct[df_correct['response_time'] > 0].dropna(subset=['response_time'])

    # 2. Visualizacion
    # Verificar la distribución de los tiempos de respuesta
    plt.figure(figsize=(15, 6))
    sns.histplot(df_correct['response_time'], kde=True, color='blue')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Density')
    plt.title('Distribution of Response Times')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Mean response time by experience
    plt.figure(figsize=(15, 6))
    sns.barplot(x='experience', y='response_time', data=df_correct, ci=95, palette='colorblind', capsize=0.1)
    plt.xlabel('Experience')
    plt.ylabel('Mean Response Time (ms)')
    plt.title('Mean Response Times by Experience')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Mean response time by validity
    plt.figure(figsize=(15, 6))
    sns.barplot(x='valid', y='response_time', data=df_correct, ci=95, palette='colorblind', capsize=0.1)
    plt.xlabel('Validity (True/False)')
    plt.ylabel('Mean Response Time (ms)')
    plt.title('Mean Response Times by Validity')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 3. Transformacion de datos
    # Apply Box-Cox transformation to response time
    rt_data = df_correct['response_time'].dropna()
    if (rt_data <= 0).any():
        rt_data += abs(rt_data.min()) + 1
    transformed_data, lambda_opt = boxcox(rt_data)
    df_correct['response_time_boxcox'] = transformed_data

    # Comparar la distribución antes y después de la transformación
    plt.figure(figsize=(15, 6))
    sns.histplot(rt_data, kde=True, color='blue', label='Original Data')
    sns.histplot(transformed_data, kde=True, color='green', label='Box-Cox Transformed Data')
    plt.xlabel('Response Time')
    plt.ylabel('Density')
    plt.title('Distribution Before and After Box-Cox Transformation')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Q-Q plot por experiencia antes de eliminar el outlier
    plt.figure(figsize=(15, 10))
    for i, experience in enumerate(df_correct['experience'].unique()):
        plt.subplot(2, 2, i + 1)
        probplot(df_correct[df_correct['experience'] == experience]['response_time_boxcox'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot (Before Outlier Removal) - Experience: {experience}')
    plt.tight_layout()
    plt.show()

    # Q-Q plot por valid antes de eliminar el outlier
    plt.figure(figsize=(10, 5))
    for i, validity in enumerate(df_correct['valid'].unique()):
        plt.subplot(1, 2, i + 1)
        probplot(df_correct[df_correct['valid'] == validity]['response_time_boxcox'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot (Before Outlier Removal) - Valid: {validity}')
    plt.tight_layout()
    plt.show()

    # Eliminar outlier en respuestas válidas
    valid_rt = df_correct[df_correct['valid'] == True]['response_time_boxcox']
    q1 = valid_rt.quantile(0.25)
    q3 = valid_rt.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    valid_rt_no_outliers = valid_rt[(valid_rt >= lower_bound) & (valid_rt <= upper_bound)]

    df_correct_no_outliers = df_correct[df_correct['valid'] == True]
    df_correct_no_outliers = df_correct_no_outliers[(df_correct_no_outliers['response_time_boxcox'] >= lower_bound) & (df_correct_no_outliers['response_time_boxcox'] <= upper_bound)]
    df_correct_no_outliers = pd.concat([df_correct_no_outliers, df_correct[df_correct['valid'] == False]])

    # Comparar resultados antes y después de eliminar el outlier
    comparison_results = []

    # 4. Verificacion de normalidad
    # Antes de eliminar el outlier
    shapiro_stat_before, shapiro_p_value_before = shapiro(df_correct['response_time_boxcox'])
    comparison_results.append(['Shapiro-Wilk Test (Before Outlier Removal)', shapiro_stat_before, shapiro_p_value_before])
    # Después de eliminar el outlier
    shapiro_stat_after, shapiro_p_value_after = shapiro(df_correct_no_outliers['response_time_boxcox'])
    comparison_results.append(['Shapiro-Wilk Test (After Outlier Removal)', shapiro_stat_after, shapiro_p_value_after])

    # Q-Q plot por experiencia después de eliminar el outlier
    plt.figure(figsize=(15, 10))
    for i, experience in enumerate(df_correct_no_outliers['experience'].unique()):
        plt.subplot(2, 2, i + 1)
        probplot(df_correct_no_outliers[df_correct_no_outliers['experience'] == experience]['response_time_boxcox'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot (After Outlier Removal) - Experience: {experience}')
    plt.tight_layout()
    plt.show()

    # Q-Q plot por valid después de eliminar el outlier
    plt.figure(figsize=(10, 5))
    for i, validity in enumerate(df_correct_no_outliers['valid'].unique()):
        plt.subplot(1, 2, i + 1)
        probplot(df_correct_no_outliers[df_correct_no_outliers['valid'] == validity]['response_time_boxcox'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot (After Outlier Removal) - Valid: {validity}')
    plt.tight_layout()
    plt.show()

    # 5. Verificacion de homogeneidad de varianzas
    experiences = df_correct['experience'].unique()
    # Antes de eliminar el outlier
    groups_before = [df_correct[df_correct['experience'] == exp]['response_time_boxcox'] for exp in experiences]
    levene_stat_before, levene_p_value_before = levene(*groups_before)
    comparison_results.append(['Levene Test (Before Outlier Removal)', levene_stat_before, levene_p_value_before])
    # Después de eliminar el outlier
    groups_after = [df_correct_no_outliers[df_correct_no_outliers['experience'] == exp]['response_time_boxcox'] for exp in experiences]
    levene_stat_after, levene_p_value_after = levene(*groups_after)
    comparison_results.append(['Levene Test (After Outlier Removal)', levene_stat_after, levene_p_value_after])

    # 6. Aplicacion de prueba de hipotesis
    # Mann-Whitney U test between words (valid) and non-words (invalid)
    invalid_rt = df_correct[df_correct['valid'] == False]['response_time_boxcox']
    # Antes de eliminar el outlier
    if valid_rt.empty or invalid_rt.empty:
        comparison_results.append(['Mann-Whitney U Test (Before Outlier Removal)', 'N/A', 'N/A'])
    else:
        mannwhitney_stat_before, mannwhitney_p_value_before = mannwhitneyu(valid_rt, invalid_rt, alternative='two-sided')
        comparison_results.append(['Mann-Whitney U Test (Before Outlier Removal)', mannwhitney_stat_before, mannwhitney_p_value_before])

    # Después de eliminar el outlier
    if valid_rt_no_outliers.empty or invalid_rt.empty:
        comparison_results.append(['Mann-Whitney U Test (After Outlier Removal)', 'N/A', 'N/A'])
    else:
        mannwhitney_stat_after, mannwhitney_p_value_after = mannwhitneyu(valid_rt_no_outliers, invalid_rt, alternative='two-sided')
        comparison_results.append(['Mann-Whitney U Test (After Outlier Removal)', mannwhitney_stat_after, mannwhitney_p_value_after])

    # Mostrar tabla comparativa de resultados
    comparison_df = pd.DataFrame(comparison_results, columns=['Test', 'Statistic', 'P-value'])
    print(comparison_df)

    # Visualizar la diferencia en tiempos de respuesta sin el outlier
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_correct_no_outliers, x='valid', y='response_time_boxcox', palette='colorblind')
    plt.xlabel('Validity')
    plt.ylabel('Box-Cox Transformed Response Time (without outliers)')
    plt.title('Response Time Distribution by Validity without Outliers')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Compare effect sizes across experiences
    print("\nComparison of Effect Sizes Across Experiences:")
    for result in comparison_results:
        print(result)

    # 8. Estimacion de efectos
    if mannwhitney_stat_after is not None:
        valid_n_after = len(valid_rt_no_outliers)
        invalid_n_after = len(invalid_rt)
        if valid_n_after > 0 and invalid_n_after > 0:
            effect_size_after = (2 * mannwhitney_stat_after) / (valid_n_after * invalid_n_after) - 1
            print("\nEffect Size (Rank-Biserial Correlation) After Outlier Removal:")
            print(f"Effect Size: {effect_size_after:.4f}")
        else:
            print("Error: No se pudo calcular el tamaño del efecto debido a la falta de datos.")
    else:
        print("Error: No se pudo calcular el tamaño del efecto debido a resultados faltantes en el Mann-Whitney U Test.")


# Imprimir detalles del Mann-Whitney U Test (After Outlier Removal)
print("\nDetalles del Mann-Whitney U Test (After Outlier Removal):")
print(f"U-Statistic: {mannwhitney_stat_after}")
print(f"P-Value: {mannwhitney_p_value_after}")

# Aplicar la prueba de Kruskal-Wallis para comparar las medianas entre grupos (por experiencia)
kruskal_stat, kruskal_p_value = kruskal(*[df_correct_no_outliers[df_correct_no_outliers['experience'] == exp]['response_time_boxcox'] for exp in experiences])

# Agregar el resultado a la tabla de comparación
comparison_results.append(['Kruskal-Wallis Test (After Outlier Removal)', kruskal_stat, kruskal_p_value])

# Imprimir detalles del Kruskal-Wallis Test
print("\nDetalles del Kruskal-Wallis Test (After Outlier Removal):")
print(f"Statistic: {kruskal_stat}")
print(f"P-Value: {kruskal_p_value}")

# Actualizar e imprimir la tabla comparativa de resultados
comparison_df = pd.DataFrame(comparison_results, columns=['Test', 'Statistic', 'P-value'])
print("\nTabla Actualizada de Resultados de Pruebas de Hipótesis Aplicadas:")
print(comparison_df)

# Aplicar la prueba de Kruskal-Wallis para evaluar la interacción entre 'experience' y 'valid'
interaction_groups = [df_correct_no_outliers[(df_correct_no_outliers['experience'] == exp) & (df_correct_no_outliers['valid'] == valid)]['response_time_boxcox']
                      for exp in experiences for valid in df_correct_no_outliers['valid'].unique()]

# Verificar que cada grupo tenga datos antes de realizar la prueba
interaction_groups = [group for group in interaction_groups if len(group) > 0]

kruskal_stat_interaction, kruskal_p_value_interaction = kruskal(*interaction_groups)

# Agregar el resultado a la tabla de comparación
comparison_results.append(['Kruskal-Wallis Test (Interaction: Experience x Valid)', kruskal_stat_interaction, kruskal_p_value_interaction])

# Imprimir detalles del Kruskal-Wallis Test para la interacción
print("\nDetalles del Kruskal-Wallis Test para la Interacción (Experience x Valid):")
print(f"Statistic: {kruskal_stat_interaction}")
print(f"P-Value: {kruskal_p_value_interaction}")

# Actualizar e imprimir la tabla comparativa de resultados
comparison_df = pd.DataFrame(comparison_results, columns=['Test', 'Statistic', 'P-value'])
print("\nTabla Actualizada de Resultados de Pruebas de Hipótesis Aplicadas (incluyendo interacción):")
print(comparison_df)

# Actualizar la tabla comparativa de resultados con las columnas adicionales requeridas
comparison_results_with_variables = [
    ['Shapiro-Wilk Test (Before Outlier Removal)', 'response_time_boxcox', shapiro_stat_before, shapiro_p_value_before],
    ['Shapiro-Wilk Test (After Outlier Removal)', 'response_time_boxcox', shapiro_stat_after, shapiro_p_value_after],
    ['Levene Test (Before Outlier Removal)', 'experience', levene_stat_before, levene_p_value_before],
    ['Kruskal-Wallis Test (After Outlier Removal)', 'experience', kruskal_stat, kruskal_p_value],
    ['Kruskal-Wallis Test (Interaction: Experience x Valid)', 'experience x valid', kruskal_stat_interaction, kruskal_p_value_interaction]
]

# Crear el DataFrame con las columnas solicitadas
comparison_df_with_variables = pd.DataFrame(comparison_results_with_variables, columns=['Test', 'Variables', 'Statistic', 'P-value'])

# Imprimir la tabla actualizada de resultados de pruebas de hipótesis aplicadas
print("\nTabla Actualizada de Resultados de Pruebas de Hipótesis Aplicadas (con Variables):")
print(comparison_df_with_variables)

# Agregar tamaño del efecto a la tabla de comparación
effect_sizes = []

# Calcular tamaños del efecto para los tests aplicables
# Para el Mann-Whitney U Test (después de eliminar outliers)
if mannwhitney_stat_after is not None:
    valid_n_after = len(valid_rt_no_outliers)
    invalid_n_after = len(invalid_rt)
    if valid_n_after > 0 and invalid_n_after > 0:
        effect_size_after = (2 * mannwhitney_stat_after) / (valid_n_after * invalid_n_after) - 1
        effect_sizes.append(effect_size_after)
    else:
        effect_sizes.append('N/A')
else:
    effect_sizes.append('N/A')

# Para el Kruskal-Wallis Test para experiencia
if kruskal_stat > 0:
    eta_squared_kruskal = kruskal_stat / (len(df_correct_no_outliers) - 1)
    effect_sizes.append(eta_squared_kruskal)
else:
    effect_sizes.append('N/A')

# Para la interacción experiencia x valid (Kruskal-Wallis)
if kruskal_stat_interaction > 0:
    eta_squared_interaction = kruskal_stat_interaction / (len(df_correct_no_outliers) - 1)
    effect_sizes.append(eta_squared_interaction)
else:
    effect_sizes.append('N/A')

# Añadir tamaños del efecto a la tabla comparativa
for i in range(len(comparison_results_with_variables)):
    if len(effect_sizes) > i:
        comparison_results_with_variables[i].append(effect_sizes[i])
    else:
        comparison_results_with_variables[i].append('N/A')

# Crear el DataFrame actualizado con la columna de tamaño del efecto
comparison_df_with_variables_effect_size = pd.DataFrame(comparison_results_with_variables, columns=['Test', 'Variables', 'Statistic', 'P-value', 'Effect Size'])

# Imprimir la tabla actualizada de resultados de pruebas de hipótesis aplicadas
print("\nTabla Actualizada de Resultados de Pruebas de Hipótesis Aplicadas (con Tamaño del Efecto):")
print(comparison_df_with_variables_effect_size)

from scikit_posthocs import posthoc_dunn

# Aplicar contrastes post-hoc usando Dunn's Test para la variable 'experience'
posthoc_experience = posthoc_dunn(df_correct_no_outliers, val_col='response_time_boxcox', group_col='experience', p_adjust='bonferroni')

# Imprimir resultados del test post-hoc Dunn para 'experience'
print("\nResultados de Dunn's Post-Hoc Test para 'experience' (Exp_1 vs Exp_0):")
print(posthoc_experience)

# Aplicar contrastes post-hoc para la interacción 'experience x valid'
df_correct_no_outliers['interaction'] = df_correct_no_outliers['experience'] + " x " + df_correct_no_outliers['valid'].astype(str)
posthoc_interaction = posthoc_dunn(df_correct_no_outliers, val_col='response_time_boxcox', group_col='interaction', p_adjust='bonferroni')

# Imprimir resultados del test post-hoc Dunn para la interacción 'experience x valid'
print("\nResultados de Dunn's Post-Hoc Test para la Interacción 'experience x valid':")
print(posthoc_interaction)

# Calcular el tamaño del efecto para las comparaciones post-hoc
# Utilizaremos la correlación de rangos biserial para las comparaciones significativas
effect_sizes_posthoc = []

# Calcular tamaño del efecto para 'experience' (Exp_1 vs Exp_0)
if posthoc_experience.loc['Exp_1', 'Exp_0'] < 0.05:  # Significativo
    group1 = df_correct_no_outliers[df_correct_no_outliers['experience'] == 'Exp_1']['response_time_boxcox']
    group2 = df_correct_no_outliers[df_correct_no_outliers['experience'] == 'Exp_0']['response_time_boxcox']
    u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    effect_size_experience = (2 * u_stat) / (len(group1) * len(group2)) - 1
    effect_sizes_posthoc.append(['experience (Exp_1 vs Exp_0)', effect_size_experience])

# Calcular tamaño del efecto para la interacción 'experience x valid'
for interaction in posthoc_interaction.columns:
    if posthoc_interaction.loc[interaction, interaction] < 0.05:  # Significativo
        group1 = df_correct_no_outliers[df_correct_no_outliers['interaction'] == interaction]['response_time_boxcox']
        group2 = df_correct_no_outliers[df_correct_no_outliers['interaction'] != interaction]['response_time_boxcox']
        u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
        effect_size_interaction = (2 * u_stat) / (len(group1) * len(group2)) - 1
        effect_sizes_posthoc.append([f'interaction ({interaction})', effect_size_interaction])

# Imprimir tamaños del efecto calculados para las interacciones significativas
print("\nTamaños del Efecto para Comparaciones Post-Hoc Significativas:")
for effect in effect_sizes_posthoc:
    print(f"{effect[0]}: Effect Size = {effect[1]:.4f}")

# Aplicar contrastes post-hoc manualmente usando Mann-Whitney U Test para la variable 'experience'
experience_groups = df_correct_no_outliers['experience'].unique()
posthoc_results_experience = []

for comb in combinations(experience_groups, 2):
    group1 = df_correct_no_outliers[df_correct_no_outliers['experience'] == comb[0]]['response_time_boxcox']
    group2 = df_correct_no_outliers[df_correct_no_outliers['experience'] == comb[1]]['response_time_boxcox']
    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    posthoc_results_experience.append([f'{comb[0]} vs {comb[1]}', stat, p_value])

# Imprimir resultados del test post-hoc para 'experience' (Exp_1 vs Exp_0)
print("\nResultados de Mann-Whitney U Test para Comparaciones Post-Hoc de 'experience':")
for result in posthoc_results_experience:
    print(f"{result[0]}: U-Statistic = {result[1]:.4f}, P-Value = {result[2]:.4e}")

# Calcular el tamaño del efecto para las comparaciones significativas
effect_sizes_posthoc_experience = []

for result in posthoc_results_experience:
    if result[2] < 0.05:  # Significativo
        group1 = df_correct_no_outliers[df_correct_no_outliers['experience'] == result[0].split(' vs ')[0]]['response_time_boxcox']
        group2 = df_correct_no_outliers[df_correct_no_outliers['experience'] == result[0].split(' vs ')[1]]['response_time_boxcox']
        u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
        effect_size = (2 * u_stat) / (len(group1) * len(group2)) - 1
        effect_sizes_posthoc_experience.append([result[0], effect_size])

# Imprimir tamaños del efecto calculados para comparaciones significativas
print("\nTamaños del Efecto para Comparaciones Post-Hoc Significativas (Experience):")
for effect in effect_sizes_posthoc_experience:
    print(f"{effect[0]}: Effect Size = {effect[1]:.4f}")

# Comparar 'experience' x 'valid' con Mann-Whitney U Test
validity_groups = df_correct_no_outliers['valid'].unique()
posthoc_results_interaction = []

for exp in experience_groups:
    for valid in validity_groups:
        group1 = df_correct_no_outliers[(df_correct_no_outliers['experience'] == exp) & (df_correct_no_outliers['valid'] == valid)]['response_time_boxcox']
        for exp2 in experience_groups:
            for valid2 in validity_groups:
                if exp != exp2 or valid != valid2:
                    group2 = df_correct_no_outliers[(df_correct_no_outliers['experience'] == exp2) & (df_correct_no_outliers['valid'] == valid2)]['response_time_boxcox']
                    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    posthoc_results_interaction.append([f'({exp}, {valid}) vs ({exp2}, {valid2})', stat, p_value])

# Imprimir resultados del test post-hoc para la interacción 'experience x valid'
print("\nResultados de Mann-Whitney U Test para Comparaciones Post-Hoc de la Interacción 'experience x valid':")
for result in posthoc_results_interaction:
    if result[2] < 0.05:  # Mostrar solo los resultados significativos
        print(f"{result[0]}: U-Statistic = {result[1]:.4f}, P-Value = {result[2]:.4e}")
