import re
import textwrap
from inspect import getsource, getmodule
from string import Template

HANDLER_TEMPLATE=Template("""
from ${base_class_module} import ${base_class}

class ${handler_name}(${base_class}):
${init}
${preprocess}
${inference}
${postprocess}
""")

HANDLER_METHODS = [
    "init",
    "initialize",
    "preprocess",
    "postprocess",
]

DEPLOY_KWARGS = [
    "handler_name",
]


POSSIBLE_KWARGS = HANDLER_METHODS + DEPLOY_KWARGS


def deploy(func):
    _check_deployable(func)
    
    print(_codegen_handler_from_deployable(func))
    
    
    
    
def deployable(base_class, **kwargs):
    print(f"{kwargs=}")
    def inner(func, **inner_kwargs):
        func.__is_deployable__ = True
        func._kwargs = kwargs
        func._base_class = base_class
        
        return func
    
    return inner


def _codegen_handler_from_deployable(dep):
    
    inference_source = getsource(dep)
    pattern = r'\@deployable\([^)]+\)'
    inference_source = re.sub(pattern, '', inference_source)
    
    sub_dict = {
        "handler_name": dep._kwargs.get("handle_name", "CustomHandler"),
        "base_class": dep._base_class.__name__,
        "base_class_module": getmodule(dep._base_class).__name__,
        "inference": _correct_indentation(inference_source),
    }
    for f_name in HANDLER_METHODS:
        func = dep._kwargs.get(f_name, None)
        sub_dict[f_name] = _correct_indentation(getsource(func)) if func else ""
        
    handler_code = HANDLER_TEMPLATE.substitute(**sub_dict)
            
    return handler_code


def _check_deployable(func):
    assert hasattr(func, "__is_deployable__")
    assert hasattr(func, "_kwargs")
    assert hasattr(func, "_base_class")
    unknown_kwargs = [kw for kw in func._kwargs.keys() if kw not in POSSIBLE_KWARGS]
    assert len(unknown_kwargs)==0, f"Unknown parameter given: {unknown_kwargs}"
    

def _correct_indentation(source):
        source = textwrap.dedent(source)
        source = textwrap.indent(source, '    ')
        return source



# def _create_mar_from_deployable(dep):
    

    
    
    
    