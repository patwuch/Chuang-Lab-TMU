#!/usr/bin/env nextflow

/*
 * Refined Nextflow DSL2 pipeline for Dengue XGBoost workflow.
 * Modules: prepare_data, optuna_tuning, final_train_eval
 * Outputs are organized under `results/` relative to project root.
 */

nextflow.enable.dsl=2

// ---------------------------
// PARAMETERS
// ---------------------------
params.config_input = "WT1014"
params.n_trials = 40
params.project_root = "../"  // adjust to your repo root
params.results_dir = "${params.project_root}/results"

// ---------------------------
// MODULE: PREPARE_DATA
// ---------------------------
process PREPARE_DATA {
    tag "prepare_data"

    input:
        val config_input   // <- no `.from()` here

    output:
        path "prepared_data" dir:true

    script:
    """
    python3 ${params.project_root}/scripts/prepare_data.py \
        --config_input ${config_input} \
        --project_root ${params.project_root} \
        --outdir prepared_data
    """
}


// ---------------------------
// MODULE: OPTUNA_TUNING
// ---------------------------
process OPTUNA_TUNING {
    tag "optuna_tuning"

    publishDir "${params.results_dir}/optuna", mode: 'copy'

    input:
        path prepared from PREPARE_DATA
        val n_trials from params.n_trials

    output:
        path "optuna_results" dir:true

    script:
    """
    mkdir -p optuna_results
    python3 ${params.project_root}/scripts/optuna_tuning.py \
        --prepared_dir ${prepared} \
        --n_trials ${n_trials} \
        --outdir optuna_results
    """
}

// ---------------------------
// MODULE: FINAL_TRAIN_EVAL
// ---------------------------
process FINAL_TRAIN_EVAL {
    tag "final_train_eval"

    publishDir "${params.results_dir}/final", mode: 'copy'

    input:
        path prepared from PREPARE_DATA
        path tuned from OPTUNA_TUNING

    output:
        path "final_outputs" dir:true

    script:
    """
    mkdir -p final_outputs
    python3 ${params.project_root}/scripts/final_train_eval.py \
        --prepared_dir ${prepared} \
        --optuna_dir ${tuned} \
        --outdir final_outputs
    """
}

// ---------------------------
// WORKFLOW
// ---------------------------
workflow {
    prepared_data_ch = PREPARE_DATA(params.config_input)
    optuna_results_ch = OPTUNA_TUNING(prepared_data_ch, params.n_trials)
    FINAL_TRAIN_EVAL(prepared_data_ch, optuna_results_ch)
}