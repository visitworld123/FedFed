import sys

from yacs.config import CfgNode as _CfgNode

sys.path.append('..')

class CfgNode(_CfgNode):

    def setup(self, args):
        if args.config_file is not None:
            self.merge_from_file((args.config_file))
        if args.opts is not None:
            self.merge_from_list(args.opts)

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            v = f"'{v}'" if isinstance(v, str) else v
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

global_cfg = CfgNode()
CN = CfgNode

def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    from .default import _C
    return _C.clone()
