use std::collections::{BTreeMap, BTreeSet};

use ndarray::{ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString};

use crate::py_file::WrfFile;
use wrf_formula::{
    parse_unit, Axis, BoundaryPolicy,
    CompiledFormula as NativeCompiledFormula, ErrorKind, EvaluationOptions,
    FormulaError as NativeFormulaError, FormulaOutput, GridConvention, GridLocation,
    HeightDatum, MissingPolicy, NonFinitePolicy, ParameterSpec, ParameterValues, Recipe,
    RecipeReference, RecipeRequirements, Requirement, ResourceLimits,
};

create_exception!(_wrf, FormulaError, PyException);
create_exception!(_wrf, FormulaSyntaxError, FormulaError);
create_exception!(_wrf, FormulaNameError, FormulaError);
create_exception!(_wrf, FormulaUnitError, FormulaError);
create_exception!(_wrf, FormulaShapeError, FormulaError);
create_exception!(_wrf, FormulaResourceError, FormulaError);
create_exception!(_wrf, FormulaEvaluationError, FormulaError);

const MAX_SOURCE_BYTES: usize = 64 * 1024;
const MAX_METADATA_ITEMS: usize = 1024;
const MAX_METADATA_BYTES: usize = 256 * 1024;
const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_UNIT_BYTES: usize = 256;

#[pyclass(name = "_CompiledFormula")]
struct CompiledFormula {
    inner: NativeCompiledFormula,
    name: Option<String>,
    description: Option<String>,
    expected_units: Option<String>,
    evaluation_options: EvaluationOptions,
}

#[pymethods]
impl CompiledFormula {
    #[getter]
    fn source(&self) -> String {
        self.inner.source().to_string()
    }

    #[getter]
    fn canonical_source(&self) -> String {
        self.inner.canonical_source().to_string()
    }

    #[getter]
    fn dependencies(&self) -> Vec<String> {
        self.inner.plan().dependencies.clone()
    }

    /// Return compile-only metadata. This never opens or reads a WRF file.
    fn plan(&self, py: Python<'_>) -> PyResult<PyObject> {
        let plan = self.inner.plan();
        let result = PyDict::new(py);
        result.set_item("canonical_source", &plan.canonical_source)?;
        result.set_item("dependencies", &plan.dependencies)?;
        result.set_item("functions", &plan.functions)?;
        result.set_item("assignments", &plan.assignments)?;
        result.set_item(
            "requirements",
            plan.requirements
                .iter()
                .map(requirement_name)
                .collect::<Vec<_>>(),
        )?;
        result.set_item("ast_nodes", plan.ast_nodes)?;
        result.set_item("ast_depth", plan.ast_depth)?;
        if let Some(requirements) = &plan.recipe_requirements {
            result.set_item(
                "recipe_requirements",
                recipe_requirements_to_object(py, requirements)?,
            )?;
        }
        let (uses_horizontal_calculus, uses_vertical_derivative) =
            plan_calculus_usage(plan);
        if uses_horizontal_calculus && uses_vertical_derivative {
            result.set_item(
                "calculus_convention",
                "horizontal: WRF mass-point conformal projected derivatives; vertical: physical-height derivatives along fixed model columns; not the strict raw C-grid AVO/PVO/UH stencil",
            )?;
        } else if uses_horizontal_calculus {
            result.set_item(
                "calculus_convention",
                "WRF mass-point conformal projected derivatives; not the strict raw C-grid AVO/PVO/UH stencil",
            )?;
        } else if uses_vertical_derivative {
            result.set_item(
                "calculus_convention",
                "physical-height derivatives along fixed terrain-following model columns",
            )?;
        }
        let warnings = if uses_horizontal_calculus {
            vec![
                "generic Formula Lab calculus uses the WRF mass-grid convention and is not strict NCAR raw C-grid AVO/PVO/UH parity",
            ]
        } else {
            Vec::new()
        };
        result.set_item("warnings", warnings)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature = (wrffile, *, timeidx=0, parameters=None, boundary_policy=None, missing_policy=None, non_finite_policy=None, variable_unit_overrides=None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        wrffile: &WrfFile,
        timeidx: usize,
        parameters: Option<&Bound<'_, PyDict>>,
        boundary_policy: Option<&str>,
        missing_policy: Option<&str>,
        non_finite_policy: Option<&str>,
        variable_unit_overrides: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let parameter_values = extract_parameter_values(parameters)?;
        let mut evaluation_options = self.evaluation_options.clone();
        if let Some(value) = boundary_policy {
            evaluation_options.boundary_policy = parse_boundary_policy(value)?;
        }
        if let Some(value) = missing_policy {
            evaluation_options.missing_policy = parse_missing_policy(value)?;
        }
        if let Some(value) = non_finite_policy {
            evaluation_options.non_finite_policy = parse_non_finite_policy(value)?;
        }
        if let Some(overrides) = variable_unit_overrides {
            evaluation_options
                .variable_unit_overrides
                .extend(extract_unit_overrides(overrides)?);
        }

        // Intentionally retain the GIL. wrf-core's single-time cache currently
        // has more than one mutex and must first adopt one proven lock order
        // before Python may expose concurrent evaluation safely.
        let output = self
            .inner
            .evaluate_wrf(
                wrffile.inner(),
                timeidx,
                &parameter_values,
                &evaluation_options,
            )
            .map_err(|error| formula_error(py, error))?;

        if let Some(expected_units) = &self.expected_units {
            validate_expected_units(py, expected_units, &output.units)?;
        }
        output_to_python(
            py,
            output,
            self.name.as_deref(),
            self.description.as_deref(),
        )
    }

    fn __repr__(&self) -> String {
        format!("_CompiledFormula({:?})", self.inner.canonical_source())
    }
}

#[pyfunction(name = "_compile_formula")]
fn compile_formula(py: Python<'_>, recipe: &Bound<'_, PyAny>) -> PyResult<CompiledFormula> {
    let recipe = extract_recipe(recipe)?;
    let name = recipe.name.clone();
    let description = (!recipe.description.trim().is_empty())
        .then(|| recipe.description.clone());
    let expected_units = recipe.expected_output_units.clone();
    let evaluation_options = recipe.evaluation_options.clone();
    let inner = recipe
        .compile()
        .map_err(|error| formula_error(py, error))?;
    Ok(CompiledFormula {
        inner,
        name: Some(name),
        description,
        expected_units,
        evaluation_options,
    })
}

fn extract_recipe(value: &Bound<'_, PyAny>) -> PyResult<Recipe> {
    let recipe = value.downcast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("native _compile_formula expects a validated recipe dictionary")
    })?;
    reject_unknown_keys(
        recipe,
        &[
            "schema",
            "name",
            "version",
            "description",
            "authors",
            "references",
            "tags",
            "source",
            "parameters",
            "expected_output_units",
            "requirements",
            "evaluation_options",
            "resource_limits",
        ],
        "recipe",
    )?;

    let schema = required_string(recipe, "schema")?;
    let name = required_string(recipe, "name")?;
    let version = required_string(recipe, "version")?;
    let description = optional_string(recipe, "description")?.unwrap_or_default();
    let authors = optional_string_list(recipe, "authors")?.unwrap_or_default();
    let references = match recipe.get_item("references")? {
        Some(value) => extract_references(&value)?,
        None => Vec::new(),
    };
    let tags = optional_string_list(recipe, "tags")?.unwrap_or_default();
    let source = required_bounded_string(recipe, "source", MAX_SOURCE_BYTES)?;
    let parameters = match recipe.get_item("parameters")? {
        Some(value) => extract_parameter_specs(&value)?,
        None => Vec::new(),
    };
    let expected_output_units = optional_string(recipe, "expected_output_units")?;
    let requirements = match recipe.get_item("requirements")? {
        Some(value) => extract_requirements(&value)?,
        None => RecipeRequirements::default(),
    };
    let evaluation_options = match recipe.get_item("evaluation_options")? {
        Some(value) => extract_evaluation_options(&value)?,
        None => EvaluationOptions::default(),
    };
    let resource_limits = match recipe.get_item("resource_limits")? {
        Some(value) if !value.is_none() => Some(extract_resource_limits(&value)?),
        _ => None,
    };
    Ok(Recipe {
        schema,
        name,
        version,
        description,
        authors,
        references,
        tags,
        source,
        parameters,
        expected_output_units,
        requirements,
        evaluation_options,
        resource_limits,
    })
}

fn extract_parameter_specs(value: &Bound<'_, PyAny>) -> PyResult<Vec<ParameterSpec>> {
    let parameters = value.downcast::<PyList>().map_err(|_| {
        PyTypeError::new_err("formula recipe 'parameters' must be a list")
    })?;
    if parameters.len() > MAX_METADATA_ITEMS {
        return Err(FormulaResourceError::new_err(format!(
            "formula recipe has {} parameters; maximum is {MAX_METADATA_ITEMS}",
            parameters.len()
        )));
    }
    let mut result = Vec::with_capacity(parameters.len());
    let mut metadata_bytes = 0_usize;
    for specification in parameters.iter() {
        let specification = specification.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("formula parameter specification must be a dictionary")
        })?;
        reject_unknown_keys(
            specification,
            &["name", "default", "units", "minimum", "maximum", "description"],
            "parameter specification",
        )?;
        let name = required_bounded_string(
            specification,
            "name",
            MAX_IDENTIFIER_BYTES,
        )?;
        let units = required_bounded_string(specification, "units", MAX_UNIT_BYTES)?;
        let description = optional_string(specification, "description")?.unwrap_or_default();
        metadata_bytes = metadata_bytes
            .saturating_add(name.len())
            .saturating_add(units.len())
            .saturating_add(description.len());
        if metadata_bytes > MAX_METADATA_BYTES {
            return Err(FormulaResourceError::new_err(
                "formula parameter metadata exceeds 256 KiB",
            ));
        }
        result.push(ParameterSpec {
            name,
            units,
            default: required_f64(specification, "default")?,
            minimum: optional_f64(specification, "minimum")?,
            maximum: optional_f64(specification, "maximum")?,
            description,
        });
    }
    result.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(result)
}

fn optional_string_list(
    values: &Bound<'_, PyDict>,
    name: &str,
) -> PyResult<Option<Vec<String>>> {
    let Some(value) = values.get_item(name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let values = value.downcast::<PyList>().map_err(|_| {
        PyTypeError::new_err(format!("{name:?} must be a list of strings"))
    })?;
    if values.len() > MAX_METADATA_ITEMS {
        return Err(FormulaResourceError::new_err(format!(
            "{name:?} has {} items; maximum is {MAX_METADATA_ITEMS}",
            values.len()
        )));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut bytes = 0_usize;
    for value in values.iter() {
        let value = bounded_string(&value, name, MAX_METADATA_BYTES)?;
        bytes = bytes.saturating_add(value.len());
        if bytes > MAX_METADATA_BYTES {
            return Err(FormulaResourceError::new_err(format!(
                "{name:?} text exceeds 256 KiB"
            )));
        }
        result.push(value);
    }
    Ok(Some(result))
}

fn extract_references(value: &Bound<'_, PyAny>) -> PyResult<Vec<RecipeReference>> {
    let references = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("recipe references must be a list"))?;
    if references.len() > MAX_METADATA_ITEMS {
        return Err(FormulaResourceError::new_err(format!(
            "recipe has {} references; maximum is {MAX_METADATA_ITEMS}",
            references.len()
        )));
    }
    let mut result = Vec::with_capacity(references.len());
    let mut bytes = 0_usize;
    for item in references.iter() {
        let item = item.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("each formula reference must be a dictionary")
        })?;
        reject_unknown_keys(item, &["citation", "doi", "url"], "reference")?;
        let citation = optional_string(item, "citation")?.unwrap_or_default();
        let doi = optional_string(item, "doi")?;
        let url = optional_string(item, "url")?;
        bytes = bytes
            .saturating_add(citation.len())
            .saturating_add(doi.as_ref().map_or(0, String::len))
            .saturating_add(url.as_ref().map_or(0, String::len));
        if bytes > MAX_METADATA_BYTES {
            return Err(FormulaResourceError::new_err(
                "formula reference metadata exceeds 256 KiB",
            ));
        }
        result.push(RecipeReference { citation, doi, url });
    }
    Ok(result)
}

fn extract_requirements(value: &Bound<'_, PyAny>) -> PyResult<RecipeRequirements> {
    let values = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("recipe requirements must be a dictionary"))?;
    reject_unknown_keys(
        values,
        &[
            "fields",
            "maximum_cadence_seconds",
            "maximum_horizontal_spacing_m",
            "minimum_vertical_levels",
            "notes",
        ],
        "requirements",
    )?;
    Ok(RecipeRequirements {
        fields: optional_string_list(values, "fields")?.unwrap_or_default(),
        maximum_cadence_seconds: optional_f64(values, "maximum_cadence_seconds")?,
        maximum_horizontal_spacing_m: optional_f64(
            values,
            "maximum_horizontal_spacing_m",
        )?,
        minimum_vertical_levels: optional_usize(values, "minimum_vertical_levels")?,
        notes: optional_string_list(values, "notes")?.unwrap_or_default(),
    })
}

fn extract_evaluation_options(value: &Bound<'_, PyAny>) -> PyResult<EvaluationOptions> {
    let values = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("evaluation_options must be a dictionary"))?;
    reject_unknown_keys(
        values,
        &[
            "boundary_policy",
            "missing_policy",
            "non_finite_policy",
            "variable_unit_overrides",
        ],
        "evaluation_options",
    )?;
    let boundary_policy = parse_boundary_policy(&required_string(values, "boundary_policy")?)?;
    let missing_policy = parse_missing_policy(&required_string(values, "missing_policy")?)?;
    let non_finite_policy =
        parse_non_finite_policy(&required_string(values, "non_finite_policy")?)?;
    let variable_unit_overrides = match values.get_item("variable_unit_overrides")? {
        Some(value) => {
            let values = value.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("variable_unit_overrides must be a dictionary")
            })?;
            extract_unit_overrides(values)?
        }
        None => BTreeMap::new(),
    };
    Ok(EvaluationOptions {
        boundary_policy,
        missing_policy,
        non_finite_policy,
        variable_unit_overrides,
    })
}

fn extract_resource_limits(value: &Bound<'_, PyAny>) -> PyResult<ResourceLimits> {
    let values = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("resource_limits must be a dictionary"))?;
    reject_unknown_keys(
        values,
        &[
            "max_source_bytes",
            "max_tokens",
            "max_ast_nodes",
            "max_ast_depth",
            "max_identifier_bytes",
            "max_function_arity",
            "max_assignments",
            "max_dependencies",
            "max_output_elements",
            "max_working_bytes",
            "max_total_allocated_bytes",
            "max_operations",
        ],
        "resource_limits",
    )?;
    Ok(ResourceLimits {
        max_source_bytes: required_usize(values, "max_source_bytes")?,
        max_tokens: required_usize(values, "max_tokens")?,
        max_ast_nodes: required_usize(values, "max_ast_nodes")?,
        max_ast_depth: required_usize(values, "max_ast_depth")?,
        max_identifier_bytes: required_usize(values, "max_identifier_bytes")?,
        max_function_arity: required_usize(values, "max_function_arity")?,
        max_assignments: required_usize(values, "max_assignments")?,
        max_dependencies: required_usize(values, "max_dependencies")?,
        max_output_elements: required_usize(values, "max_output_elements")?,
        max_working_bytes: required_usize(values, "max_working_bytes")?,
        max_total_allocated_bytes: required_u64(values, "max_total_allocated_bytes")?,
        max_operations: required_u64(values, "max_operations")?,
    })
}

fn extract_parameter_values(
    values: Option<&Bound<'_, PyDict>>,
) -> PyResult<ParameterValues> {
    let Some(values) = values else {
        return Ok(ParameterValues::new());
    };
    if values.len() > MAX_METADATA_ITEMS {
        return Err(FormulaResourceError::new_err(format!(
            "formula evaluation has {} parameter overrides; maximum is {MAX_METADATA_ITEMS}",
            values.len()
        )));
    }
    let mut result = BTreeMap::new();
    for (name, value) in values.iter() {
        let name = bounded_string(&name, "formula parameter name", MAX_IDENTIFIER_BYTES)?;
        if value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "formula parameter {name:?} must be numeric, not bool"
            )));
        }
        let number = value.extract::<f64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "formula parameter {name:?} must be a real numeric scalar"
            ))
        })?;
        if !number.is_finite() {
            return Err(PyValueError::new_err(format!(
                "formula parameter {name:?} must be finite"
            )));
        }
        if result.insert(name.clone(), number).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate formula parameter {name:?}"
            )));
        }
    }
    Ok(result)
}

fn extract_unit_overrides(values: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, String>> {
    if values.len() > MAX_METADATA_ITEMS {
        return Err(FormulaResourceError::new_err(format!(
            "variable_unit_overrides has {} entries; maximum is {MAX_METADATA_ITEMS}",
            values.len()
        )));
    }
    let mut result = BTreeMap::new();
    let mut metadata_bytes = 0_usize;
    for (name, units) in values.iter() {
        let name = bounded_string(&name, "unit override name", MAX_IDENTIFIER_BYTES)?;
        let units = bounded_string(&units, "unit override", MAX_UNIT_BYTES)?;
        if name.is_empty() || units.trim().is_empty() {
            return Err(PyValueError::new_err(
                "unit override names and unit strings must not be empty",
            ));
        }
        metadata_bytes = metadata_bytes
            .saturating_add(name.len())
            .saturating_add(units.len());
        if metadata_bytes > MAX_METADATA_BYTES {
            return Err(FormulaResourceError::new_err(
                "variable_unit_overrides metadata exceeds 256 KiB",
            ));
        }
        result.insert(name, units);
    }
    Ok(result)
}

fn reject_unknown_keys(
    values: &Bound<'_, PyDict>,
    allowed: &[&str],
    context: &str,
) -> PyResult<()> {
    if values.len() > allowed.len() {
        return Err(PyValueError::new_err(format!(
            "{context} has {} keys; maximum known key count is {}",
            values.len(),
            allowed.len()
        )));
    }
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    for (key, _) in values.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        if !allowed.contains(key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown {context} key {key:?}"
            )));
        }
    }
    Ok(())
}

fn required_string(values: &Bound<'_, PyDict>, name: &str) -> PyResult<String> {
    required_bounded_string(values, name, MAX_METADATA_BYTES)
}

fn required_bounded_string(
    values: &Bound<'_, PyDict>,
    name: &str,
    max_bytes: usize,
) -> PyResult<String> {
    let value = values
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required key {name:?}")))?;
    bounded_string(&value, name, max_bytes)
}

fn optional_string(values: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    optional_bounded_string(values, name, MAX_METADATA_BYTES)
}

fn optional_bounded_string(
    values: &Bound<'_, PyDict>,
    name: &str,
    max_bytes: usize,
) -> PyResult<Option<String>> {
    let Some(value) = values.get_item(name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    bounded_string(&value, name, max_bytes).map(Some)
}

fn bounded_string(value: &Bound<'_, PyAny>, name: &str, max_bytes: usize) -> PyResult<String> {
    let value = value
        .downcast::<PyString>()
        .map_err(|_| PyTypeError::new_err(format!("{name:?} must be a string")))?;
    let text = value.to_str()?;
    if text.len() > max_bytes {
        return Err(FormulaResourceError::new_err(format!(
            "{name:?} is {} UTF-8 bytes; maximum is {max_bytes}",
            text.len()
        )));
    }
    Ok(text.to_string())
}

fn required_f64(values: &Bound<'_, PyDict>, name: &str) -> PyResult<f64> {
    let value = values
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required key {name:?}")))?;
    finite_f64(&value, name)?.ok_or_else(|| {
        PyValueError::new_err(format!("missing required numeric value {name:?}"))
    })
}

fn optional_f64(values: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<f64>> {
    let Some(value) = values.get_item(name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    finite_f64(&value, name)
}

fn optional_usize(values: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<usize>> {
    let Some(value) = values.get_item(name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!("{name:?} must not be bool")));
    }
    value
        .extract::<usize>()
        .map(Some)
        .map_err(|_| PyTypeError::new_err(format!("{name:?} must be a nonnegative integer")))
}

fn required_usize(values: &Bound<'_, PyDict>, name: &str) -> PyResult<usize> {
    let value = values
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required key {name:?}")))?;
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!("{name:?} must not be bool")));
    }
    value
        .extract::<usize>()
        .map_err(|_| PyTypeError::new_err(format!("{name:?} must be a nonnegative integer")))
}

fn required_u64(values: &Bound<'_, PyDict>, name: &str) -> PyResult<u64> {
    let value = values
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required key {name:?}")))?;
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!("{name:?} must not be bool")));
    }
    value
        .extract::<u64>()
        .map_err(|_| PyTypeError::new_err(format!("{name:?} must be a nonnegative integer")))
}

fn finite_f64(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<f64>> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!("{name:?} must not be bool")));
    }
    let value = value
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err(format!("{name:?} must be a real number")))?;
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{name:?} must be finite")));
    }
    Ok(Some(value))
}

fn validate_expected_units(py: Python<'_>, expected: &str, actual: &str) -> PyResult<()> {
    let expected_unit = parse_unit(expected).map_err(|error| formula_error(py, error))?;
    let actual_unit = parse_unit(actual).map_err(|error| formula_error(py, error))?;
    if expected_unit.dimension != actual_unit.dimension
        || expected_unit.temperature_kind != actual_unit.temperature_kind
        || expected_unit.logarithmic != actual_unit.logarithmic
    {
        return Err(formula_error(
            py,
            NativeFormulaError::new(
                ErrorKind::Unit,
                format!("formula output units {actual:?} are incompatible with expected units {expected:?}"),
            ),
        ));
    }
    Ok(())
}

fn output_to_python(
    py: Python<'_>,
    output: FormulaOutput,
    recipe_name: Option<&str>,
    recipe_description: Option<&str>,
) -> PyResult<(PyObject, PyObject)> {
    let expected = output
        .shape
        .iter()
        .try_fold(1_usize, |count, size| count.checked_mul(*size))
        .ok_or_else(|| FormulaEvaluationError::new_err("formula output shape overflow"))?;
    if expected != output.data.len() || output.axes.len() != output.shape.len() {
        return Err(FormulaEvaluationError::new_err(
            "native Formula Lab returned inconsistent data, shape, or axes",
        ));
    }
    let array = ArrayD::from_shape_vec(IxDyn(&output.shape), output.data)
        .map_err(|error| FormulaEvaluationError::new_err(error.to_string()))?;
    let array = array.into_pyarray(py).into_any().unbind();

    let metadata = PyDict::new(py);
    metadata.set_item("units", &output.units)?;
    metadata.set_item(
        "description",
        recipe_description.unwrap_or(output.description.as_str()),
    )?;
    metadata.set_item(
        "axes",
        output.axes.iter().map(axis_name).collect::<Vec<_>>(),
    )?;

    let provenance = PyDict::new(py);
    provenance.set_item("engine_version", &output.provenance.engine_version)?;
    provenance.set_item("canonical_source", &output.provenance.canonical_source)?;
    provenance.set_item("source_fingerprint", &output.provenance.source_fingerprint)?;
    provenance.set_item("parameters", &output.provenance.parameters)?;
    provenance.set_item("boundary_policy", boundary_policy_name(output.provenance.boundary_policy))?;
    provenance.set_item("missing_policy", missing_policy_name(output.provenance.missing_policy))?;
    provenance.set_item(
        "non_finite_policy",
        non_finite_policy_name(output.provenance.non_finite_policy),
    )?;
    provenance.set_item(
        "grid_convention",
        output
            .provenance
            .grid_convention
            .as_ref()
            .map(grid_convention_name),
    )?;
    provenance.set_item(
        "vertical_height_datums",
        output
            .provenance
            .vertical_height_datums
            .iter()
            .map(height_datum_name)
            .collect::<Vec<_>>(),
    )?;
    provenance.set_item("warnings", &output.provenance.warnings)?;
    provenance.set_item("base_time_index", output.provenance.base_time_index)?;
    provenance.set_item("valid_time", &output.provenance.valid_time)?;
    provenance.set_item("input_identity", &output.provenance.input_identity)?;
    provenance.set_item("native_recipe_name", &output.provenance.recipe_name)?;
    provenance.set_item("native_recipe_version", &output.provenance.recipe_version)?;
    provenance.set_item("recipe_references", &output.provenance.recipe_references)?;
    if let Some(requirements) = &output.provenance.recipe_requirements {
        provenance.set_item(
            "recipe_requirements",
            recipe_requirements_to_object(py, requirements)?,
        )?;
    }
    provenance.set_item(
        "variable_unit_overrides",
        &output.provenance.variable_unit_overrides,
    )?;
    if let Some(name) = recipe_name {
        provenance.set_item("recipe_name", name)?;
    }

    let py_inputs = pyo3::types::PyList::empty(py);
    for input in &output.provenance.inputs {
        let item = PyDict::new(py);
        item.set_item("requested_name", &input.requested_name)?;
        item.set_item("resolved_name", &input.resolved_name)?;
        item.set_item("time_offset", input.time_offset)?;
        item.set_item("shape", &input.shape)?;
        item.set_item(
            "axes",
            input.axes.iter().map(axis_name).collect::<Vec<_>>(),
        )?;
        item.set_item("source_units", &input.source_units)?;
        item.set_item("effective_units", &input.effective_units)?;
        item.set_item("unit_override_used", &input.unit_override_used)?;
        item.set_item("grid_location", grid_location_name(&input.grid_location))?;
        py_inputs.append(item)?;
    }
    provenance.set_item("inputs", py_inputs)?;
    metadata.set_item("provenance", provenance)?;
    Ok((array, metadata.into_any().unbind()))
}

fn recipe_requirements_to_object(
    py: Python<'_>,
    requirements: &RecipeRequirements,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("fields", &requirements.fields)?;
    result.set_item(
        "maximum_cadence_seconds",
        requirements.maximum_cadence_seconds,
    )?;
    result.set_item(
        "maximum_horizontal_spacing_m",
        requirements.maximum_horizontal_spacing_m,
    )?;
    result.set_item(
        "minimum_vertical_levels",
        requirements.minimum_vertical_levels,
    )?;
    result.set_item("notes", &requirements.notes)?;
    Ok(result.into_any().unbind())
}

fn formula_error(py: Python<'_>, error: NativeFormulaError) -> PyErr {
    let result = match error.kind {
        ErrorKind::Limit => FormulaResourceError::new_err(error.message.clone()),
        ErrorKind::Lex | ErrorKind::Parse => FormulaSyntaxError::new_err(error.message.clone()),
        ErrorKind::UnknownIdentifier
        | ErrorKind::UnknownFunction
        | ErrorKind::Arity
        | ErrorKind::Parameter => FormulaNameError::new_err(error.message.clone()),
        ErrorKind::Unit => FormulaUnitError::new_err(error.message.clone()),
        ErrorKind::Shape | ErrorKind::Grid => FormulaShapeError::new_err(error.message.clone()),
        ErrorKind::Compile => FormulaError::new_err(error.message.clone()),
        ErrorKind::Time
        | ErrorKind::MissingData
        | ErrorKind::NonFinite
        | ErrorKind::Domain
        | ErrorKind::Resolver
        | ErrorKind::Internal => FormulaEvaluationError::new_err(error.message.clone()),
        ErrorKind::Unsupported => FormulaError::new_err(error.message.clone()),
    };
    let value = result.value(py);
    let _ = value.setattr("kind", error_kind_name(error.kind));
    let _ = value.setattr("notes", error.notes.clone());
    if let Some(span) = error.span {
        let _ = value.setattr("span", (span.start, span.end));
        let _ = value.setattr("start", span.start);
        let _ = value.setattr("end", span.end);
    } else {
        let _ = value.setattr("span", py.None());
        let _ = value.setattr("start", py.None());
        let _ = value.setattr("end", py.None());
    }
    result
}

fn error_kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Limit => "limit",
        ErrorKind::Lex => "lex",
        ErrorKind::Parse => "parse",
        ErrorKind::Compile => "compile",
        ErrorKind::UnknownIdentifier => "unknown_identifier",
        ErrorKind::UnknownFunction => "unknown_function",
        ErrorKind::Arity => "arity",
        ErrorKind::Parameter => "parameter",
        ErrorKind::Unit => "unit",
        ErrorKind::Shape => "shape",
        ErrorKind::Grid => "grid",
        ErrorKind::Time => "time",
        ErrorKind::MissingData => "missing_data",
        ErrorKind::NonFinite => "non_finite",
        ErrorKind::Domain => "domain",
        ErrorKind::Resolver => "resolver",
        ErrorKind::Unsupported => "unsupported",
        ErrorKind::Internal => "internal",
    }
}

fn plan_calculus_usage(plan: &wrf_formula::ExecutionPlan) -> (bool, bool) {
    let horizontal = plan.functions.iter().any(|name| {
        matches!(
            name.as_str(),
            "ddx" | "ddy" | "grad" | "div" | "curl" | "laplacian"
        )
    });
    let vertical = plan.functions.iter().any(|name| name == "ddz");
    (horizontal, vertical)
}

fn requirement_name(requirement: &Requirement) -> String {
    match requirement {
        Requirement::Field { name } => format!("field:{name}"),
        Requirement::MassMapFactor => "mass_map_factor".to_string(),
        Requirement::PhysicalHeight { datum } => {
            format!("physical_height:{}", height_datum_name(datum))
        }
        Requirement::AdjacentTimes => "adjacent_times".to_string(),
        Requirement::GridProjectedVector => "grid_projected_vector".to_string(),
    }
}

fn height_datum_name(datum: &HeightDatum) -> &'static str {
    match datum {
        HeightDatum::Msl => "msl",
        HeightDatum::Agl => "agl",
        HeightDatum::ExplicitField => "explicit_field",
        HeightDatum::ResolverDefault => "resolver_default",
    }
}

fn parse_boundary_policy(value: &str) -> PyResult<BoundaryPolicy> {
    match value {
        "one_sided_second_order" => Ok(BoundaryPolicy::OneSidedSecondOrder),
        "missing" => Ok(BoundaryPolicy::Missing),
        "error" => Ok(BoundaryPolicy::Error),
        _ => Err(PyValueError::new_err(format!("invalid boundary_policy {value:?}"))),
    }
}

fn parse_missing_policy(value: &str) -> PyResult<MissingPolicy> {
    match value {
        "propagate" => Ok(MissingPolicy::Propagate),
        "error" => Ok(MissingPolicy::Error),
        "ignore_in_reductions" => Ok(MissingPolicy::IgnoreInReductions),
        _ => Err(PyValueError::new_err(format!("invalid missing_policy {value:?}"))),
    }
}

fn parse_non_finite_policy(value: &str) -> PyResult<NonFinitePolicy> {
    match value {
        "propagate" => Ok(NonFinitePolicy::Propagate),
        "error" => Ok(NonFinitePolicy::Error),
        _ => Err(PyValueError::new_err(format!("invalid non_finite_policy {value:?}"))),
    }
}

fn boundary_policy_name(value: BoundaryPolicy) -> &'static str {
    match value {
        BoundaryPolicy::OneSidedSecondOrder => "one_sided_second_order",
        BoundaryPolicy::Missing => "missing",
        BoundaryPolicy::Error => "error",
    }
}

fn missing_policy_name(value: MissingPolicy) -> &'static str {
    match value {
        MissingPolicy::Propagate => "propagate",
        MissingPolicy::Error => "error",
        MissingPolicy::IgnoreInReductions => "ignore_in_reductions",
    }
}

fn non_finite_policy_name(value: NonFinitePolicy) -> &'static str {
    match value {
        NonFinitePolicy::Propagate => "propagate",
        NonFinitePolicy::Error => "error",
    }
}

fn axis_name(axis: &Axis) -> &'static str {
    match axis {
        Axis::Component => "component",
        Axis::Time => "time",
        Axis::Z => "bottom_top",
        Axis::Y => "south_north",
        Axis::X => "west_east",
    }
}

fn grid_location_name(location: &GridLocation) -> &'static str {
    match location {
        GridLocation::Mass => "mass",
        GridLocation::XFace => "x_face",
        GridLocation::YFace => "y_face",
        GridLocation::ZFace => "z_face",
        GridLocation::Unknown => "unknown",
    }
}

fn grid_convention_name(convention: &GridConvention) -> &'static str {
    match convention {
        GridConvention::WrfMassPointProjected => "wrf_mass_point_projected",
        GridConvention::Cartesian => "cartesian",
    }
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FormulaError", py.get_type::<FormulaError>())?;
    m.add("FormulaSyntaxError", py.get_type::<FormulaSyntaxError>())?;
    m.add("FormulaNameError", py.get_type::<FormulaNameError>())?;
    m.add("FormulaUnitError", py.get_type::<FormulaUnitError>())?;
    m.add("FormulaShapeError", py.get_type::<FormulaShapeError>())?;
    m.add("FormulaResourceError", py.get_type::<FormulaResourceError>())?;
    m.add(
        "FormulaEvaluationError",
        py.get_type::<FormulaEvaluationError>(),
    )?;
    m.add_class::<CompiledFormula>()?;
    m.add_function(wrap_pyfunction!(compile_formula, m)?)?;
    Ok(())
}
