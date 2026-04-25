# Отчёт за 2026-04-15

## 1) Что сделали сегодня

1. Сформировали и обновили блоки базовой динамики: RMSD и RMSF по системам/мутациям.
2. Собрали кластеризацию RMSF по ранам и матрицу согласованности ранов.
3. Подготовили агрегированные дельта-графики по мутантам: delta RMSF, delta DCCM и локальные H-bond узлы.
4. Выполнили блоки helicity, hbond lifetimes и phospho-tail coupling (включая SASA-распределения и explanation-панели).
5. Выполнили финальную интерпретацию route-selective/asymmetric gating в компактном формате для презентации.
6. Выполнили отдельный ICL2-анализ по двум протомерным маршрутам (B и R) для систем A/B/C.
7. Оценили участие моста 665-670 в маршрутах и асимметрию прохождения сигнала.
8. Построили локальное сравнение DCCM вдоль путей для системы B.
9. Экспортировали структуры для VMD с beta-кодировкой путей и gate-области.
10. Построили график regulation penalty для A/B/C с явным акцентом на систему B.
11. Построили общий сравнительный барчарт для WT + мутантов (B vs R): A, B, C, C665G, I666G, A667G, R668G, I669G, F670G, TM3-6G.

## 2) Ключевые результаты

### 2.1 ICL2 асимметрия (A/B/C)

- A: cost_B = 2.602, cost_R = 3.005, delta(R-B) = +0.403
- B: cost_B = 3.069, cost_R = 3.498, delta(R-B) = +0.429
- C: cost_B = 2.951, cost_R = 3.545, delta(R-B) = +0.594

Вывод: в A/B/C маршрут protomer_B устойчиво эффективнее (меньшая стоимость пути), чем protomer_R.

### 2.2 Мост 665-670 (по ICL2 summary)

- В системе B для пути protomer_R зафиксирован транзит через мост 665-670 (bridge_used = true, bridge_count = 1).
- Для остальных путей в A/B/C bridge_used = false.
- Интерпретация: участок 665-670 работает как регулируемый gate/транзитный узел, а не обязательный «разрыв» пути.

### 2.3 Локальный DCCM по системе B

- protomer_B: local_dccm_mean_edge = 0.8605
- protomer_R: local_dccm_mean_edge = 0.8794

(Профили сравнивались по рёбрам найденных путей; итоговая эффективность трактуется через интегральную стоимость пути.)

### 2.4 WT + мутанты (B vs R)

- A: delta(R-B) = +0.403
- B: delta(R-B) = +0.429
- C: delta(R-B) = +0.594
- C665G: delta(R-B) = -0.189
- I666G: delta(R-B) = +0.025
- A667G: delta(R-B) = -0.060
- R668G: delta(R-B) = -0.334
- I669G: delta(R-B) = -0.335
- F670G: delta(R-B) = +1.104
- TM3-6G: delta(R-B) = -0.341

Вывод: мутации в зоне gate в основном перестраивают асимметрию маршрутов (route regulation), а не дают единый сценарий полного блокирования обоих путей.

## 3) Какие графики есть (полный список)

Всего в results_4/mfti_2026_ru_v2: 67 графических файлов.

По разделам:
- rmsd: 9
- rmsf: 20
- 13_icl2_asymmetry: 4
- 12_signal_path: 2
- 08_phospho_tail_coupling: 24
- 06_helicity: 1
- 07_hbond: 1
- корневые итоговые панели: 6

### 3.1 RMSD (все)

- results_4/mfti_2026_ru_v2/rmsd/rmsd_A_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_A667G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_C665G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_F670G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_I666G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_I669G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_mglu3_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_R668G_runs.png
- results_4/mfti_2026_ru_v2/rmsd/rmsd_TM3-6G_runs.png

### 3.2 RMSF (все)

- results_4/mfti_2026_ru_v2/rmsf/rmsf_A_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_A_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_A667G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_A667G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_B_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_B_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_C665G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_C665G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_F670G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_F670G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_I666G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_I666G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_I669G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_I669G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_mglu3_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_mglu3_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_R668G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_R668G_runs.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_TM3-6G_run_correlation.png
- results_4/mfti_2026_ru_v2/rmsf/rmsf_TM3-6G_runs.png

### 3.3 Остальные графики после RMSD/RMSF

- results_4/mfti_2026_ru_v2/rmsf_run_clustering.png
- results_4/mfti_2026_ru_v2/01_delta_rmsf_mutants_vs_wt_ru_all.png
- results_4/mfti_2026_ru_v2/02_delta_dccm_mutants_vs_wt_ru_all.png
- results_4/mfti_2026_ru_v2/03_local_hbonds_665_667_ru.png
- results_4/mfti_2026_ru_v2/04_mglu3_no_arrestin_dccm_ru.png
- results_4/mfti_2026_ru_v2/09_mglu3_no_arrestin_dccm_ru.png
- results_4/mfti_2026_ru_v2/06_helicity/06_helicity_loop_mutants_vs_wt_ru.png
- results_4/mfti_2026_ru_v2/07_hbond/07_hbond_mglu3_arrestin_lifetimes_top10_ru.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/sasa_focus_heatmap_ru.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/A/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/A/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/A667G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/A667G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/C665G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/C665G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/F670G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/F670G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/I666G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/I666G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/I669G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/I669G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/mglu3/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/mglu3/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/R668G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/R668G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/TM3-6G/T_to_A/01_sasa/T_phospho_residue_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/TM3-6G/T_to_A/01_sasa/T_phospho_total_sasa_distribution.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/A_minus_mutants_signed_delta_sasa.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/kde_pS856_sasa_A_vs_mutants.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/kde_pS857_sasa_A_vs_mutants.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/kde_pS859_sasa_A_vs_mutants.png
- results_4/mfti_2026_ru_v2/08_phospho_tail_coupling/explanation/kde_pT860_sasa_A_vs_mutants.png
- results_4/mfti_2026_ru_v2/12_signal_path/connectedness_vs_mobility_scatter.png
- results_4/mfti_2026_ru_v2/12_signal_path/final_system_similarity_heatmap.png
- results_4/mfti_2026_ru_v2/13_icl2_asymmetry/all_systems_comparison_bar.png
- results_4/mfti_2026_ru_v2/13_icl2_asymmetry/g_signal_efficiency_by_protomer.png
- results_4/mfti_2026_ru_v2/13_icl2_asymmetry/regulation_penalty_bar.png
- results_4/mfti_2026_ru_v2/13_icl2_asymmetry/system_B_local_dccm_compare.png

Полный технический реестр также сохранён в файле:
- results_4/mfti_2026_ru_v2/ALL_PLOTS_LIST.txt

## 4) Дополнительные артефакты (структуры/данные)

- PDB:
  - results_4/mfti_2026_ru_v2/13_icl2_asymmetry/mglu3_asymmetric_g_paths.pdb
  - results_4/mfti_2026_ru_v2/13_icl2_asymmetry/mglu3_asymmetric_gate.pdb
- JSON:
  - results_4/mfti_2026_ru_v2/13_icl2_asymmetry/icl2_asymmetry_summary.json
  - results_4/mfti_2026_ru_v2/13_icl2_asymmetry/asymmetric_gate_summary.json
  - results_4/mfti_2026_ru_v2/13_icl2_asymmetry/all_systems_comparison_bar_data.json
  - results_4/mfti_2026_ru_v2/12_signal_path/final_mutant_comparison.json
  - results_4/mfti_2026_ru_v2/12_signal_path/summary_signal_path.json

---

Отчёт собран по фактически присутствующим артефактам в результатах и по выполненным сегодня вычислениям/визуализациям.
