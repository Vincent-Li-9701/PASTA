"""
    This file defines the Hierarchy of Graph Tree class and PartNet data loader.
"""

import os
import json
import torch
import numpy as np
from torch.utils import data
from .quaternion import Quaternion
from collections import namedtuple
from pytorch3d import transforms

# store a part hierarchy of graphs for a shape
class Tree(object):

    # global object category information
    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = None
    root_sem = None

    @ staticmethod
    def load_category_info(cat):
        with open(os.path.join('../stats/part_semantics/', cat+'.txt'), 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if '/' in y:
                    Tree.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        Tree.num_sem = len(Tree.part_name2id) + 1
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)
        Tree.root_sem = Tree.part_id2name[1]


    # store a part node in the tree
    class Node(object):

        def __init__(self, part_id=0, is_leaf=False, box=None, label=None, children=None, edges=None, full_label=None, geo=None, geo_feat=None):
            self.is_leaf = is_leaf          # store True if the part is a leaf node
            self.part_id = part_id          # part_id in result_after_merging.json of PartNet
            self.box = box                  # box parameter for all nodes
            self.geo = geo                  # 1 x 1000 x 3 point cloud
            self.geo_feat = geo_feat        # 1 x 100 geometry feature
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            self.children = [] if children is None else children
                                            # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges
                                            # all of its children relationships; 
                                            # each entry is a tuple <part_a, part_b, type, params, dist>
            """
                Here defines the edges format:
                    part_a, part_b:
                        Values are the order in self.children (e.g. 0, 1, 2, 3, ...).
                        This is an directional edge for A->B.
                        If an edge is commutative, you may need to manually specify a B->A edge.
                        For example, an ADJ edge is only shown A->B, 
                        there is no edge B->A in the json file.
                    type:
                        Four types considered in StructureNet: ADJ, ROT_SYM, TRANS_SYM, REF_SYM.
                    params:
                        There is no params field for ADJ edge;
                        For ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle;
                        For TRANS_SYM edge, 0-2 translation vector;
                        For REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers, 
                            3-5 unit normal direction of the reflection plane.
                    dist:
                        For ADJ edge, it's the closest distance between two parts;
                        For SYM edge, it's the chamfer distance after matching part B to part A.
            """
        
        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]
            
        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            out[0, Tree.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)
            
        def get_box_quat(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir /= np.linalg.norm(xdir)
            ydir = box[9:]
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
            rotmat = np.vstack([xdir, ydir, zdir]).T
            q = Quaternion(matrix=rotmat)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            box_quat = np.hstack([center, size, quat]).astype(np.float32)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)

        def get_part_hierarchy(self, leafs_only=False):
            part_boxes = []
            part_ids = []
            part_sems = []

            # STUDENT CODE START
            if (leafs_only and self.is_leaf) or (not leafs_only):
                part_boxes.append(self.box)
                part_ids.append(self.part_id)
                part_sems.append(self.full_label)
            
            for child in self.children:
                _part_boxes, _part_ids, _part_sems = child.get_part_hierarchy(leafs_only)
                part_boxes += _part_boxes
                part_ids += _part_ids
                part_sems += _part_sems
           
            # STUDENT CODE END

            return part_boxes, part_ids, part_sems
            
        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.cpu().numpy().squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
            rotmat = q.rotation_matrix
            box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1)
            
        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)

            for child_node in self.children:
                child_node.to(device)

            return self

        def _to_str(self, level, pid, detailed=False):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}'
            if detailed:
                out_str += 'Box('+';'.join([str(item) for item in self.box.numpy()])+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)

            if detailed and len(self.edges) > 0:
                for edge in self.edges:
                    if 'params' in edge:
                        edge = edge.copy() # so the original parameters don't get changed
                        edge['params'] = edge['params'].cpu().numpy()
                    out_str += '  |'*(level) + '  ├' + 'Edge(' + str(edge) + ')\n'

            return out_str

        def __str__(self):
            return self._to_str(0, 0)

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        def child_adjacency(self, typed=False, max_children=None):
            if max_children is None:
                adj = torch.zeros(len(self.children), len(self.children))
            else:
                adj = torch.zeros(max_children, max_children)

            if typed:
                edge_types = ['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM']

            for edge in self.edges:
                if typed:
                    edge_type_index = edge_types.index(edge['type'])
                    adj[edge['part_a'], edge['part_b']] = edge_type_index
                    adj[edge['part_b'], edge['part_a']] = edge_type_index
                else:
                    adj[edge['part_a'], edge['part_b']] = 1
                    adj[edge['part_b'], edge['part_a']] = 1

            return adj

        def geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_geos = []; out_nodes = []
            for node in nodes:
                if not leafs_only or node.is_leaf:
                    out_geos.append(node.geo)
                    out_nodes.append(node)
            return out_geos, out_nodes

        def boxes(self, per_node=False, leafs_only=False):
            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.box)

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes
            
        def graph(self, leafs_only=False):
            part_boxes = []
            part_geos = []
            part_angles = []
            part_quats = []
            edges = []
            part_ids = []
            part_sems = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue

                    part_boxes.append(child.get_box_quat().squeeze(0))
                    part_geos.append(child.geo)
                    part_ids.append(child.part_id)
                    part_sems.append(child.label)
                    quat = child.get_box_quat().squeeze(0)[6:]
                    axis_angles = transforms.quaternion_to_axis_angle(quat)
                    part_angles.append(axis_angles)
                    part_quats.append(quat)

                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                for edge in node.edges:
                    if leafs_only and not (
                            node.children[edge['part_a']].is_leaf and
                            node.children[edge['part_b']].is_leaf):
                        continue
                    edges.append(edge.copy())
                    edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]

                box_index_offset += child_count

            return part_boxes, part_geos, part_angles, part_quats, part_sems

        def edge_tensors(self, edge_types, device, type_onehot=True):
            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor(
                [[e['part_a'], e['part_b']] for e in self.edges] + [[e['part_b'], e['part_a']] for e in self.edges],
                device=device, dtype=torch.long).view(1, num_edges*2, 2)

            # get edge type as tensor
            edge_type = torch.tensor([edge_types.index(edge['type']) for edge in self.edges], device=device, dtype=torch.long)
            if type_onehot:
                edge_type = one_hot(inp=edge_type, label_count=len(edge_types)).transpose(0, 1).view(1, num_edges, len(edge_types)).to(dtype=torch.float32)
            else:
                edge_type = edge_type.view(1, num_edges)
            edge_type = torch.cat([edge_type, edge_type], dim=1) # add edges in other direction (symmetric adjacency)

            return edge_type, edge_indices

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt


    # functions for class Tree
    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def depth_first_traversal(self):
        return self.root.depth_first_traversal()

    def boxes(self, per_node=False, leafs_only=False):
        return self.root.boxes(per_node=per_node, leafs_only=leafs_only)

    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    def get_part_hierarchy(self, leafs_only=False):
        return self.root.get_part_hierarchy(leafs_only=leafs_only)

    def free(self):
        for node in self.depth_first_traversal():
            del node.geo
            del node.geo_feat
            del node.box
            del node


# extend torch.data.Dataset class for PartNet
class PartNetDataset(data.Dataset):

    def __init__(self, root, label_dict, bin_centers, object_list, data_features, \
            load_geo=False, load_points=False, load_clip=False, bounds=None):
        self.root = root
        self.label_dict = label_dict
        self.bin_centers = bin_centers
        self.data_features = data_features
        self.load_geo = load_geo
        self.load_points = load_points
        self.load_clip = load_clip

        print(os.path.join(self.root, object_list))

        #TODO: remove this from the code 
        # below objects contains invalid tree structure
        problems = ['12233', '12268', '12675', '12784', '37076', '47142', '48492', '46178', '47208', \
                    '46701', '47414', '46023', '44854', '46195', '47053', '48687', '45928', \
                    '6003', '4385', '11309', '47865', '45128', '47951', '12139', '49085', '17238', \
                    '15044', '14455', '17504', '16566', '17091', '14514', '17131', '14568', '16832', \
                    '15480', '14948', '17301', '14380', '15809', '15619', '17561', '16836', '16762', \
                    '16692', '14227', '16650', '14669', '14924', '14486', '14751', '17167', '17487', \
                    '15000', '15302', '16891', '16110', '14623', '15437', '14590', '17135', '16947', \
                    '17017', '16648', '16351', '15709', '15303', '16917', '16420', '16138', '15548', \
                    '16872', '15058', '16096', '15555', '14621', '15066', '14913', '15579', '14943', \
                    '14313', '14655', '16507', '15396', '16814', '14538', '17180', '17111', '14841', \
                    '16628', '16377', '14395', '15316', '15130', '16487', '15216', '15913', '14883', \
                    '16572', '16418', '14830', '17473', '17592', '16969', '15048', '14461', '15346', \
                    '16088', '17560', '16706', '16677', '17340', '14374', '15062', '15263', '17113', \
                    '14440', '16700', '15283', '17793', '15666', '17604', '14688']
        
        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
            self.object_names = [item for item in self.object_names if item not in problems]
        else:
            self.object_names = object_list

        if bounds is None:
            self._compute_bounds()
        else:
            self._part_boxes_bounds = bounds['part_boxes']
            self._root_box_bounds = bounds['part_boxes'] #bounds['root_box']

    def __getitem__(self, index):
        if 'object' in self.data_features:
            obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'), \
                    load_geo=self.load_geo, load_points=self.load_points, load_clip=self.load_clip)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat
                
        return data_feats

    def __len__(self):
        return len(self.object_names)

    def _compute_bounds(self):
        _centroid_min = np.array([10000000]*3).astype('float32')
        _centroid_max = np.array([-10000000]*3).astype('float32')
        _size_min = np.array([10000000]*3).astype('float32')
        _size_max = np.array([-10000000]*3).astype('float32')

        _root_centroid_min = np.array([10000000]*3).astype('float32')
        _root_centroid_max = np.array([-10000000]*3).astype('float32')
        _root_size_min = np.array([10000000]*3).astype('float32')
        _root_size_max = np.array([-10000000]*3).astype('float32')

        for idx in range(self.__len__()):
            tree = self.load_object(os.path.join(self.root, self.object_names[idx]+'.json'), \
                    load_geo=self.load_geo, load_points=self.load_points, load_clip=self.load_clip)
            
            part_boxes, _, _, part_quats, _ = tree.graph(leafs_only=True)
            part_boxes, part_quats = torch.stack(part_boxes), torch.stack(part_quats)

            part_centers = part_boxes[:, :3]
            center_maxs = torch.max(part_centers, dim=0)[0] #ignoring the indices max values
            center_mins = torch.min(part_centers, dim=0)[0]
            part_sizes = part_boxes[:, 3:6]
            size_maxs = torch.max(part_sizes, dim=0)[0]
            size_mins = torch.min(part_sizes, dim=0)[0]

            _centroid_max = np.maximum(center_maxs, _centroid_max)
            _centroid_min = np.minimum(center_mins, _centroid_min)
            _size_max = np.maximum(size_maxs, _size_max)
            _size_min = np.minimum(size_mins, _size_min)

            root_box = tree.root.get_box_quat().squeeze()
            root_centers = root_box[:3] 
            root_sizes = root_box[3:6]

            _root_centroid_max = np.maximum(root_centers, _root_centroid_max)
            _root_centroid_min = np.minimum(root_centers, _root_centroid_min)
            _root_size_max = np.maximum(root_sizes, _root_size_max)
            _root_size_min = np.minimum(root_sizes, _root_size_min)

        self._part_boxes_bounds = [torch.cat([_centroid_min, _size_min]), \
            torch.cat([_centroid_max, _size_max])]
        self._root_box_bounds = [torch.cat([_root_centroid_min, _root_size_min]), \
            torch.cat([_root_centroid_max, _root_size_max])]


    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                load_geo=self.load_geo, load_points=self.load_points)
        return obj

    @property
    def bounds(self):

        return {
            'part_boxes': self._part_boxes_bounds,
            'root_box': self._root_box_bounds
        }
    
    @staticmethod
    def load_object(fn, load_geo=False, load_points=False, load_clip=False):
        if load_geo:
            geo_fn = fn.replace('_hier', '_geo').replace('json', 'npz')
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)

            if load_geo:
                node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        if load_points:
            point_cloud_fn = fn.replace('_hier', '_mesh_tsdf/points').replace('json', 'npy')
            point_label_fn = fn.replace('_hier', '_mesh_tsdf/labels').replace('json', 'npy')
            point_clouds = np.load(point_cloud_fn).astype('float32')
            point_label = np.load(point_label_fn).astype('float32')
            
            obj.point_clouds = point_clouds
            obj.point_label = point_label
        else:
            obj.point_clouds = np.array([])
            obj.point_label = np.array([])

        if load_clip:
            clip_fn = fn.replace('_hier', '_mesh/clip').replace('json', 'npy')
            clip_feature = np.load(clip_fn).astype('float32')
            obj.clip_feature = clip_feature
        else:
            obj.clip_feature = np.array([])

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.geo is not None:
                node_json['geo'] = node.geo.cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(edge)
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        with open(fn, 'w') as f:
            json.dump(obj_json, f)


# out shape: (label_count, in shape)
def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out
