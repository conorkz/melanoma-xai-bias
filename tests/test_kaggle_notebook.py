"""Tests for the generated Kaggle benchmark notebook."""

import ast
import importlib.util
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_kaggle_notebook_is_valid_and_complete(tmp_path):
    builder = load_script("build_kaggle_notebook")
    path = builder.build(tmp_path / "kaggle_benchmark.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    sources = "\n".join(cell.source for cell in notebook.cells)
    assert "conorkz/melanoma-xai-bias" in sources
    assert "sweep_stage1.yaml" in sources
    assert "sweep_stage2.yaml" in sources
    assert "collect_results.py" in sources
    assert "teacher_update.zip" in sources
    assert "val_sensitivity_mean" in sources
    assert "GPU P100" in sources

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    for cell in code_cells:
        ast.parse(cell.source)


def test_build_stage1_notebook_is_valid_and_complete(tmp_path):
    builder = load_script("build_kaggle_notebook")
    path = builder.build_stage1(tmp_path / "kaggle_stage1.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    sources = "\n".join(cell.source for cell in notebook.cells)
    assert "sweep_stage1.yaml" in sources
    assert "collect_results.py" in sources
    assert "stage1_results.zip" in sources
    assert "val_sensitivity_mean" in sources
    assert "sweep_stage2.yaml" not in sources
    assert "WINNER =" not in sources

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    for cell in code_cells:
        ast.parse(cell.source)


def test_build_stage2_notebook_is_valid_and_complete(tmp_path):
    builder = load_script("build_kaggle_notebook")
    path = builder.build_stage2(tmp_path / "kaggle_stage2.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    sources = "\n".join(cell.source for cell in notebook.cells)
    assert "sweep_stage2.yaml" in sources
    assert "collect_results.py" in sources
    assert "teacher_update.zip" in sources
    assert "sweep_stage1.yaml" not in sources
    assert "27 runs" in sources

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    for cell in code_cells:
        ast.parse(cell.source)


def test_committed_stage2_notebook_matches_builder(tmp_path):
    builder = load_script("build_kaggle_notebook")
    generated = builder.build_stage2(tmp_path / "generated_stage2.ipynb")
    committed_path = PROJECT_ROOT / "notebooks" / "kaggle_stage2.ipynb"
    assert committed_path.exists()
    generated_notebook = nbformat.read(generated, as_version=4)
    committed_notebook = nbformat.read(committed_path, as_version=4)
    assert len(generated_notebook.cells) == len(committed_notebook.cells)
    for generated_cell, committed_cell in zip(generated_notebook.cells, committed_notebook.cells):
        assert generated_cell.cell_type == committed_cell.cell_type
        assert generated_cell.source == committed_cell.source


def test_committed_stage1_notebook_matches_builder(tmp_path):
    builder = load_script("build_kaggle_notebook")
    generated = builder.build_stage1(tmp_path / "generated_stage1.ipynb")
    committed_path = PROJECT_ROOT / "notebooks" / "kaggle_stage1.ipynb"
    assert committed_path.exists()
    generated_notebook = nbformat.read(generated, as_version=4)
    committed_notebook = nbformat.read(committed_path, as_version=4)
    assert len(generated_notebook.cells) == len(committed_notebook.cells)
    for generated_cell, committed_cell in zip(generated_notebook.cells, committed_notebook.cells):
        assert generated_cell.cell_type == committed_cell.cell_type
        assert generated_cell.source == committed_cell.source


def test_committed_kaggle_notebook_matches_builder(tmp_path):
    builder = load_script("build_kaggle_notebook")
    generated = builder.build(tmp_path / "generated.ipynb")
    committed_path = PROJECT_ROOT / "notebooks" / "kaggle_benchmark.ipynb"
    assert committed_path.exists()
    generated_notebook = nbformat.read(generated, as_version=4)
    committed_notebook = nbformat.read(committed_path, as_version=4)
    assert len(generated_notebook.cells) == len(committed_notebook.cells)
    for generated_cell, committed_cell in zip(generated_notebook.cells, committed_notebook.cells):
        assert generated_cell.cell_type == committed_cell.cell_type
        assert generated_cell.source == committed_cell.source
