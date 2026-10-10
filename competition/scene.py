"""Two physical Go2s; only reset writes their poses."""
from copy import deepcopy
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def scene_digest():
    """Hash physics and meshes without depending on the checkout directory."""
    root=ET.fromstring(scene_xml())
    root.find('compiler').set('meshdir','assets')
    digest=hashlib.sha256(ET.tostring(root))
    for path in sorted((ROOT/'models/go2/assets').rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(ROOT/'models/go2')).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def scene_xml():
    source = ET.parse(ROOT / 'models/go2/go2_posctrl.xml').getroot()
    root = ET.Element('mujoco', model='rq2026_1v1')
    compiler = deepcopy(source.find('compiler'))
    compiler.set('meshdir', str(ROOT / 'models/go2/assets'))
    root.append(compiler)
    option = deepcopy(source.find('option'))
    option.set('timestep', '.002')
    root.append(option)
    root.append(deepcopy(source.find('default')))
    root.append(deepcopy(source.find('asset')))
    ET.SubElement(root, 'statistic', center='0 0 .3', extent='4')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 6', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'camera', name='top_view', pos='0 0 9', quat='1 0 0 0', fovy='45')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1',
                  rgba='.4 .5 .4 1', friction='.8 .1 .1')
    for name, pos, size in [('n','0 2.55 .5','2.6 .05 .5'),
                             ('s','0 -2.55 .5','2.6 .05 .5'),
                             ('e','2.55 0 .5','.05 2.6 .5'),
                             ('w','-2.55 0 .5','.05 2.6 .5')]:
        ET.SubElement(world, 'geom', name='wall_'+name, type='box', pos=pos,
                      size=size, rgba='.6 .6 .7 1')
    actuators = ET.SubElement(root, 'actuator')
    for role in ('flee', 'oni'):
        body = deepcopy(source.find('worldbody/body'))
        for element in body.iter():
            if 'name' in element.attrib:
                element.set('name', role+'_'+element.get('name'))
            if role == 'oni' and element.tag == 'geom' and element.get('class') == 'visual':
                element.attrib.pop('material', None)
                element.set('rgba', '.85 .15 .12 1')
        # The free joint must have an explicit role-specific name.
        body.find('freejoint').set('name', role+'_root')
        world.append(body)
        for original in source.find('actuator'):
            actuator = deepcopy(original)
            actuator.set('name', role+'_'+original.get('name'))
            actuator.set('joint', role+'_'+original.get('joint'))
            actuator.set('kp', '60')
            actuators.append(actuator)
    return ET.tostring(root, encoding='unicode')
