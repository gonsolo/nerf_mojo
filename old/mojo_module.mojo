import python
from python import PythonObject
from python import Python
from python.bindings import PythonModuleBuilder
import math
from os import abort

struct Nerfinger(Defaultable & Movable & Representable):
    var i: Int

    fn __init__(out self):
        self.i = 0
    fn __init__(out self, i: Int):
        self.i = i

    fn __moveinit__(out self, owned existing: Self):
       self.i = existing.i

    fn __repr__(self) -> String:
        return "Nerfinger: " + repr(self.i)

@export
fn PyInit_mojo_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[factorial]("factorial", docstring="Compute n!")
        m.add_type[Nerfinger]("Nerfinger")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))

fn factorial(py_obj: PythonObject) raises -> PythonObject:
    # Raises an exception if `py_obj` is not convertible to a Mojo `Int`.
    var n = Int(py_obj)

    return math.factorial(n)

