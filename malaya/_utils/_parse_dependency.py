import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from collections import defaultdict
from itertools import chain
from six import string_types


class DependencyGraph(object):
    def __init__(
        self,
        tree_str = None,
        cell_extractor = None,
        zero_based = False,
        cell_separator = None,
        top_relation_label = 'ROOT',
    ):
        self.nodes = defaultdict(
            lambda: {
                'address': None,
                'word': None,
                'lemma': None,
                'ctag': None,
                'tag': None,
                'feats': None,
                'head': None,
                'deps': defaultdict(list),
                'rel': None,
            }
        )

        self.nodes[0].update({'ctag': 'TOP', 'tag': 'TOP', 'address': 0})

        self.root = None

        if tree_str:
            self._parse(
                tree_str,
                cell_extractor = cell_extractor,
                zero_based = zero_based,
                cell_separator = cell_separator,
                top_relation_label = top_relation_label,
            )

    def remove_by_address(self, address):
        del self.nodes[address]

    def redirect_arcs(self, originals, redirect):
        for node in self.nodes.values():
            new_deps = []
            for dep in node['deps']:
                if dep in originals:
                    new_deps.append(redirect)
                else:
                    new_deps.append(dep)
            node['deps'] = new_deps

    def add_arc(self, head_address, mod_address):
        relation = self.nodes[mod_address]['rel']
        self.nodes[head_address]['deps'].setdefault(relation, [])
        self.nodes[head_address]['deps'][relation].append(mod_address)

    def connect_graph(self):
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if (
                    node1['address'] != node2['address']
                    and node2['rel'] != 'TOP'
                ):
                    relation = node2['rel']
                    node1['deps'].setdefault(relation, [])
                    node1['deps'][relation].append(node2['address'])

    def get_by_address(self, node_address):
        """Return the node with the given address."""
        return self.nodes[node_address]

    def contains_address(self, node_address):
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        return node_address in self.nodes

    def to_dot(self):
        s = 'digraph G{\n'
        s += 'edge [dir=forward]\n'
        s += 'node [shape=plaintext]\n'

        for node in sorted(self.nodes.values(), key = lambda v: v['address']):
            s += '\n%s [label="%s (%s)"]' % (
                node['address'],
                node['address'],
                node['word'],
            )
            for rel, deps in node['deps'].items():
                for dep in deps:
                    if rel is not None:
                        s += '\n%s -> %s [label="%s"]' % (
                            node['address'],
                            dep,
                            rel,
                        )
                    else:
                        s += '\n%s -> %s ' % (node['address'], dep)
        s += '\n}'

        return s

    def left_children(self, node_index):
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c < index)

    def right_children(self, node_index):
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c > index)

    def add_node(self, node):
        if not self.contains_address(node['address']):
            self.nodes[node['address']].update(node)

    def _parse(
        self,
        input_,
        cell_extractor = None,
        zero_based = False,
        cell_separator = None,
        top_relation_label = 'ROOT',
    ):
        def extract_3_cells(cells, index):
            word, tag, head = cells
            return index, word, word, tag, tag, '', head, ''

        def extract_4_cells(cells, index):
            word, tag, head, rel = cells
            return index, word, word, tag, tag, '', head, rel

        def extract_7_cells(cells, index):
            line_index, word, lemma, tag, _, head, rel = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, tag, tag, '', head, rel

        def extract_10_cells(cells, index):
            line_index, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, ctag, tag, feats, head, rel

        extractors = {
            3: extract_3_cells,
            4: extract_4_cells,
            7: extract_7_cells,
            10: extract_10_cells,
        }

        if isinstance(input_, string_types):
            input_ = (line for line in input_.split('\n'))

        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)

        cell_number = None
        for index, line in enumerate(lines, start = 1):
            cells = line.split(cell_separator)
            if cell_number is None:
                cell_number = len(cells)
            else:
                assert cell_number == len(cells)

            if cell_extractor is None:
                try:
                    cell_extractor = extractors[cell_number]
                except KeyError:
                    raise ValueError(
                        'Number of tab-delimited fields ({0}) not supported by '
                        'CoNLL(10) or Malt-Tab(4) format'.format(cell_number)
                    )

            try:
                index, word, lemma, ctag, tag, feats, head, rel = cell_extractor(
                    cells, index
                )
            except (TypeError, ValueError):
                word, lemma, ctag, tag, feats, head, rel = cell_extractor(cells)

            if head == '_':
                continue

            head = int(head)
            if zero_based:
                head += 1

            self.nodes[index].update(
                {
                    'address': index,
                    'word': word,
                    'lemma': lemma,
                    'ctag': ctag,
                    'tag': tag,
                    'feats': feats,
                    'head': head,
                    'rel': rel,
                }
            )

            # Make sure that the fake root node has labeled dependencies.
            if (cell_number == 3) and (head == 0):
                rel = top_relation_label
            self.nodes[head]['deps'][rel].append(index)

        if self.nodes[0]['deps'][top_relation_label]:
            root_address = self.nodes[0]['deps'][top_relation_label][0]
            self.root = self.nodes[root_address]
            self.top_relation_label = top_relation_label
        else:
            warnings.warn(
                "The graph doesn't contain a node "
                'that depends on the root element.'
            )

    def _word(self, node, filter = True):
        w = node['word']
        if filter:
            if w != ',':
                return w
        return w

    def triples(self, node = None):
        """
        Extract dependency triples of the form:
        ((head word, head tag), rel, (dep word, dep tag))
        """

        if not node:
            node = self.root

        head = (node['word'], node['ctag'])
        for i in sorted(chain.from_iterable(node['deps'].values())):
            dep = self.get_by_address(i)
            yield (head, dep['rel'], (dep['word'], dep['ctag']))
            for triple in self.triples(node = dep):
                yield triple

    def _hd(self, i):
        try:
            return self.nodes[i]['head']
        except IndexError:
            return None

    def _rel(self, i):
        try:
            return self.nodes[i]['rel']
        except IndexError:
            return None

    def contains_cycle(self):
        """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [3, 1, 2, 4]

        """
        distances = {}

        for node in self.nodes.values():
            for dep in node['deps']:
                key = tuple([node['address'], dep])
                distances[key] = 1

        for _ in self.nodes:
            new_entries = {}

            for pair1 in distances:
                for pair2 in distances:
                    if pair1[1] == pair2[0]:
                        key = tuple([pair1[0], pair2[1]])
                        new_entries[key] = distances[pair1] + distances[pair2]

            for pair in new_entries:
                distances[pair] = new_entries[pair]
                if pair[0] == pair[1]:
                    path = self.get_cycle_path(
                        self.get_by_address(pair[0]), pair[0]
                    )
                    return path

        return False

    def get_cycle_path(self, curr_node, goal_node_index):
        for dep in curr_node['deps']:
            if dep == goal_node_index:
                return [curr_node['address']]
        for dep in curr_node['deps']:
            path = self.get_cycle_path(
                self.get_by_address(dep), goal_node_index
            )
            if len(path) > 0:
                path.insert(0, curr_node['address'])
                return path
        return []

    def to_conll(self, style):

        if style == 3:
            template = '{word}\t{tag}\t{head}\n'
        elif style == 4:
            template = '{word}\t{tag}\t{head}\t{rel}\n'
        elif style == 10:
            template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
        else:
            raise ValueError(
                'Number of tab-delimited fields ({0}) not supported by '
                'CoNLL(10) or Malt-Tab(4) format'.format(style)
            )

        return ''.join(
            template.format(i = i, **node)
            for i, node in sorted(self.nodes.items())
            if node['tag'] != 'TOP'
        )

    def to_graphvis(self):
        try:
            from graphviz import Source
        except:
            raise Exception(
                'graphiz not installed. Please install it and try again.'
            )
        return Source(self.to_dot())

    def to_networkx(self):
        try:
            import networkx
        except:
            raise Exception(
                'networkx not installed. Please install it and try again.'
            )

        nx_nodelist = list(range(1, len(self.nodes)))
        nx_edgelist = [
            (n, self._hd(n), self._rel(n)) for n in nx_nodelist if self._hd(n)
        ]
        self.nx_labels = {}
        for n in nx_nodelist:
            self.nx_labels[n] = self.nodes[n]['word']

        g = networkx.MultiDiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)

        return g
