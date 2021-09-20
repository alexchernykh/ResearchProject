"""
nrrd_mapping represents the encoding of  annotations for the phase FIRST INCISION:
    Key: name[:8] of the NRRD, Zb: 20190917
    Value: is a dict with a structure of
        - Key: layer number
        - Value: the corresponding name of the mask
"""
nrrd_mapping = {
    "20190917": {
        "0": "first incision",
        "1": "plane first incision",
        "2": "incised area",
        "3": "instrument frenestrated bipolar forceps",
        "4": "instrument permanent cautery hook",
        "5": "instrument cadiere forceps",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
    },
    "20191015": {
        "0": "plane first incision",
        "1": "incised area",
        "2": "first incision",
        "3": "instrument frenestrated bipolar forceps",
        "4": "instrument monopolar curved scissors",
        "5": "compress",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
    },
    "20200122": {
        "0": "instrument frenestrated bipolar forceps",
        "1": "instrument permanent cautery hook",
        "2": "plane first incision",
        "3": "not needed",
        "4": "incised area",
        "5": "first incision",
        "6": "triangle upperr corner",
        "7": "triangle lower corners",
    },
    "20200123": {
        "0": "instrument frenestrated bipolar forceps",
        "1": "instrument monopolar curved scissors",
        "2": "compress",
        "3": "plane first incision",
        "4": "incised area",
        "5": "not needed",
        "6": "triangle upperr corner",
        "7": "triangle lower corners",
        "8": "first incision",
    },
    "20200302": {
        "0": "instrument frenestrated bipolar forceps",
        "1": "instrument permanent cautery hook",
        "2": "instrument cadiere forceps",
        "3": "instrument assistant",
        "4": "plane first incision",
        "5": "incised area",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
        "9": "first incision",
    },
    "20200416": {
        "0": "plane first incision",
        "1": "first incision",
        "2": "incised area",
        "3": "instrument frenestrated bipolar forceps",
        "4": "instrument permanent cautery hook",
        "5": "instrument cadiere forceps",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
    },
    "20200423": {
        "0": "instrument frenestrated bipolar forceps",
        "1": "instrument monopolar curved scissors",
        "2": "plane first incision",
        "3": "not needed",
        "4": "first incision",
        "5": "incised area",
        "6": "triangle upperr corner",
        "7": "triangle lower corners",
    },
    "20200427": {
        "0": "incised area",
        "1": "plane first incision",
        "2": "first incision",
        "3": "compress",
        "4": "instrument frenestrated bipolar forceps",
        "5": "instrument permanent cautery hook",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
    },
    "20200504": {
        "0": "compress",
        "1": "instrument permanent cautery hook",
        "2": "instrument frenestrated bipolar forceps",
        "3": "plane first incision",
        "4": "not needed",
        "5": "triangle upperr corner",
        "6": "triangle lower corners",
        "7": "first incision",
        "8": "incised area",
    },
    "20200622": {
        "0": "compress",
        "1": "instrument frenestrated bipolar forceps",
        "2": "instrument permanent cautery hook",
        "3": "plane first incision",
        "4": "incised area",
        "5": "first incision",
        "6": "not needed",
        "7": "triangle upperr corner",
        "8": "triangle lower corners",
    },
    # this is the frame from the surgeant JW
    "JWannota": {"0": "first incision"},
    "MDannota": {"0": "first incision"},
    "TWannota": {"0": "first incision"},
    "JKannota": {"0": "first incision"},
}

"""
Provides an encoding to use it as an representation to encode the layer names as integers for FIRST INCISION
"""
mapping_layers = {
    "first incision": 0,
    "plane first incision": 1,
    "incised area": 2,
    "instrument frenestrated bipolar forceps": 3,
    "instrument permanent cautery hook": 4,
    "instrument cadiere forceps": 5,
    "not needed": 6,
    "triangle upperr corner": 7,
    "triangle lower corners": 8,
    "instrument monopolar curved scissors": 9,
    "compress": 10,
    "instrument assistant": 11,
}

"""
pedickle_package_mapping represents the encoding of  annotations for the phase PEDICKLE PACKAGE:
    Key: name[:8] of the NRRD, Zb: 20190917
    Value: is a dict with a structure of
        - Key: layer number
        - Value: the corresponding name of the mask
"""
pedickle_package_mapping = {
    "20200123": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200129": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200302": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200416": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200423": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200622": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20200717": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20201002": {"0": "pedicle package", "1": "inferior mesenteric artery"},
    "20201006": {"0": "pedicle package", "1": "inferior mesenteric artery"},
}

"""
Provides an encoding to use it as an representation to encode the layer names as integers for PEDICLE PACKAGE
"""
mapping_layers_pedickle_package = {
    "pedicle package": 0,
    "inferior mesenteric artery": 1,
}

"""
mapping_vascular_dissection represents the encoding of  annotations for the phase VASCULAR DISSECION:
    Key: name[:8] of the NRRD, Zb: 20190917
    Value: is a dict with a structure of
        - Key: layer number
        - Value: the corresponding name of the mask
"""
mapping_vascular_dissection = {
    "20191011": {"0": "inferior mesenteric artery", "1": "inferior mesenteric vein"},
    "20191012": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20191013": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200330": {
        "0": "inferior mesenteric artery",
        "1": "plastic clip",
        "2": "titan clip",
    },
    "20200331": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200332": {
        "0": "inferior mesenteric vein",
        "1": "inferior mesenteric artery",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200407": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200408": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200416": {
        "0": "inferior mesenteric artery",
        "1": "plastic clip",
        "2": "titan clip",
    },
    "20200417": {
        "0": "inferior mesenteric artery",
        "1": "plastic clip",
        "2": "titan clip",
        "3": "inferior mesenteric vein",
    },
    "20200418": {
        "0": "inferior mesenteric vein",
        "1": "plastic clip",
        "2": "titan clip",
    },
    "20200427": {
        "0": "inferior mesenteric vein",
        "1": "plastic clip",
        "2": "titan clip",
    },
    "20200428": {
        "0": "plastic clip",
        "1": "titan clip",
        "2": "inferior mesenteric artery",
        "3": "inferior mesenteric vein",
    },
    "20200429": {
        "0": "plastic clip",
        "1": "titan clip",
        "2": "inferior mesenteric vein",
    },
    "20200504": {"0": "inferior mesenteric artery", "1": "inferior mesenteric vein"},
    "20200505": {
        "0": "inferior mesenteric vein",
        "1": "inferior mesenteric artery",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200506": {
        "1": "inferior mesenteric vein",
        "0": "inferior mesenteric artery",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200622": {"1": "inferior mesenteric vein", "0": "inferior mesenteric artery"},
    "20200623": {
        "3": "inferior mesenteric vein",
        "2": "inferior mesenteric artery",
        "0": "plastic clip",
        "1": "titan clip",
    },
    "20200624": {
        "0": "plastic clip",
        "1": "titan clip",
        "2": "inferior mesenteric vein",
        "3": "dissection line gerota",
        "4": "gerotas fascia",
        "5": "mesocolon",
    },
    "20200717": {"0": "inferior mesenteric artery", "1": "inferior mesenteric vein"},
    "20200718": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200719": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200727": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "titan clip",
    },
    "20200728": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "titan clip",
    },
    "20200729": {
        "0": "inferior mesenteric artery",
        "1": "inferior mesenteric vein",
        "2": "plastic clip",
        "3": "titan clip",
    },
    "20200731": {
        "0": "inferior mesenteric artery",
    },
    "20200732": {
        "0": "inferior mesenteric artery",
        "1": "plastic clip",
        "2": "titan clip",
        "3": "inferior mesenteric vein",
    },
    "20200733": {
        "0": "inferior mesenteric vein",
        "1": "plastic clip",
        "2": "titan clip",
    },
}


"""
Provides an encoding to use it as an representation to encode the layer names as integers for VASCULAR DISSECION
"""
mapping_layers_vascular_dissection = {
    "inferior mesenteric artery": 0,
    "inferior mesenteric vein": 1,
    "plastic clip": 2,
    "titan clip": 3,
    "dissection line gerota": 4,
    "gerotas fascia": 5,
    "mesocolon": 6,
}
"""
mapping_mesocolon_gerota represents the encoding of  annotations for the phase MESOCOLON GEROTA:
    Key: name[:8] of the NRRD, Zb: 20190917
    Value: is a dict with a structure of
        - Key: layer number
        - Value: the corresponding name of the mask
"""
mapping_mesocolon_gerota = {
    "20191011": {
        "0": "mesocolon",
        "1": "gerotas fascia",
        "2": "dissection line gerota",
        "3": "lesion",
        "4": "ureter",
        "5": "exploration area",
    },
    "20191206": {
        "0": "gerotas fascia",
        "1": "mesocolon",
        "2": "dissection line gerota",
        "3": "exploration area",
    },
    "20191207": {
        "0": "gerotas fascia",
        "1": "mesocolon",
        "2": "dissection line gerota",
        "3": "exploration area",
    },
    "20191208": {
        "0": "gerotas fascia",
        "1": "mesocolon",
        "2": "dissection line gerota",
        "3": "exploration area",
    },
    "20191211": {
        "0": "gerotas fascia",
        "1": "mesocolon",
        "2": "dissection line gerota",
        "3": "exploration area",
    },
    "20191212": {
        "0": "gerotas fascia",
        "1": "mesocolon",
        "2": "dissection line gerota",
        "3": "exploration area",
    },
    "20200123": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200124": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200125": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200129": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200130": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200131": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200302": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200303": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200304": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200330": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200331": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200427": {
        "0": "dissection line gerota",
        "2": "gerotas fascia",
        "1": "mesocolon",
        "3": "exploration area",
    },
    "20200428": {
        "0": "dissection line gerota",
        "1": "pancreas",
        "2": "dissection line pancreas",
        "3": "gerotas fascia",
        "4": "mesocolon",
        "5": "exploration area",
    },
    "20200429": {
        "0": "dissection line gerota",
        "1": "pancreas",
        "2": "spleen",
        "3": "dissection line pancreas",
        "4": "gerotas fascia",
        "5": "ureter",
        "6": "mesocolon",
        "7": "exploration area",
    },
    "20200504": {
        "2": "dissection line gerota",
        "0": "gerotas fascia",
        "1": "mesocolon",
        "3": "exploration area",
    },
    "20200505": {
        "4": "dissection line gerota",
        "1": "gerotas fascia",
        "0": "mesocolon",
        "2": "pancreas",
        "3": "exploration area",
    },
    "20200506": {
        "3": "dissection line gerota",
        "0": "gerotas fascia",
        "2": "mesocolon",
        "1": "pancreas",
        "4": "exploration area",
    },
    "20200622": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200623": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20200624": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20201006": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
    },
    "20201007": {
        "0": "dissection line gerota",
        "1": "gerotas fascia",
        "2": "mesocolon",
        "3": "exploration area",
        "4": "pancreas",
    },
}


"""
Provides an encoding to use it as an representation to encode the layer names as integers for MESOCOLON GEROTA
"""
mapping_layer_mesocolon_gerota = {
    "mesocolon": 0,
    "gerotas fascia": 1,
    "dissection line gerota": 2,
    "lesion": 3,
    "ureter": 4,
    "exploration area": 5,
    "spleen": 6,
    "pancreas": 7,
    "dissection line pancreas": 8,
}
"""
mapping_tme represents the encoding of  annotations for the phase TME:
    Key: name[:8] of the NRRD, Zb: 20190917
    Value: is a dict with a structure of
        - Key: layer number
        - Value: the corresponding name of the mask
"""
mapping_tme = {
    "20191001": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20191002": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20191011": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20191012": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200123": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200124": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200125": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200129": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200130": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200302": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200303": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200330": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200331": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200423": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200424": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200622": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200623": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200717": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
    },
    "20200718": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200719": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
    "20200727": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
    },
    "20200728": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
    },
    "20200729": {
        "0": "dissection plane TME",
        "1": "dissection line TME",
        "2": "rectum",
        "3": "seminal vesicles",
    },
}


"""
Provides an encoding to use it as an representation to encode the layer names as integers for TME
"""
mapping_layers_tme = {
    "dissection plane TME": 0,
    "dissection line TME": 1,
    "rectum": 2,
    "seminal vesicles": 3,
}
