"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp


# CODE FROM PREV TASK !!!!
CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'


def throw_error(check: tp.Any, error: str) -> None:
    if check:
        raise TypeError(error)


def bind_args(func_code: types.CodeType, *args: tp.Any, **kwargs: tp.Any) -> dict[str, tp.Any]:
    posonly_slice = slice(None, func_code.co_posonlyargcount)
    pos_or_kw_slice = slice(func_code.co_posonlyargcount, func_code.co_argcount)

    parsed_pos_only_args = dict(zip(func_code.co_varnames[posonly_slice], args[posonly_slice]))
    parsed_pos_args = dict(zip(func_code.co_varnames[pos_or_kw_slice], args[pos_or_kw_slice]))

    pos_or_kw_names = frozenset(func_code.co_varnames[pos_or_kw_slice])
    posonly_names = frozenset(func_code.co_varnames[posonly_slice])
    kwonly_names = frozenset(func_code.co_varnames[
        slice(func_code.co_argcount, func_code.co_argcount + func_code.co_kwonlyargcount)
    ])

    parsed_kw_args, parsed_kw_only_args, extra_kw_args = {}, {}, {}

    for name, val in kwargs.items():
        if name in pos_or_kw_names:
            parsed_kw_args[name] = val
        elif name in kwonly_names:
            parsed_kw_only_args[name] = val
        else:
            extra_kw_args[name] = val

    co_vark = bool(func_code.co_flags & CO_VARKEYWORDS)
    throw_error(extra_kw_args.keys() & posonly_names and not co_vark, ERR_POSONLY_PASSED_AS_KW)
    throw_error(args[func_code.co_argcount:] and not bool(func_code.co_flags & CO_VARARGS), ERR_TOO_MANY_POS_ARGS)
    throw_error(extra_kw_args and not co_vark, ERR_TOO_MANY_KW_ARGS)
    throw_error(parsed_pos_args.keys() & parsed_kw_args.keys(), ERR_MULT_VALUES_FOR_ARG)

    missing_kw_only = kwonly_names - parsed_kw_only_args.keys()
    missing_pos = (posonly_names | pos_or_kw_names)\
        - parsed_pos_only_args.keys() - parsed_pos_args.keys() - parsed_kw_args.keys()

    throw_error(missing_pos, ERR_MISSING_POS_ARGS)
    throw_error(missing_kw_only, ERR_MISSING_KWONLY_ARGS)

    bound = {}
    for d in [parsed_pos_only_args,
              parsed_pos_args, parsed_kw_args, parsed_kw_only_args]:
        bound.update(d)

    if bool(func_code.co_flags & CO_VARARGS):
        var_args_name = func_code.co_varnames[func_code.co_argcount + func_code.co_kwonlyargcount]
        bound[var_args_name] = args[func_code.co_argcount:]

    if bool(func_code.co_flags & CO_VARKEYWORDS):
        var_keywords_name = func_code.co_varnames[
            func_code.co_argcount + func_code.co_kwonlyargcount + bool(func_code.co_flags & CO_VARARGS)
        ]
        bound[var_keywords_name] = extra_kw_args

    return bound

# END CODE FROM PREV TASK !!!!


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None

        self.next_instr = 0
        self.kw: tp.Optional[tp.Any | None] = None

    def top(self) -> tp.Any:
        if len(self.data_stack) == 0:
            raise NameError('stack is empty')  # Костыль??
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        if len(self.data_stack) == 0:
            raise NameError('stack is empty')  # Костыль??
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))

        while self.next_instr < len(instructions):
            curr_instr_ind = self.next_instr

            instruction = instructions[curr_instr_ind]
            name = instruction.opname.lower()

            if name == 'binary_op':
                getattr(self, name + "_op")(instruction.argrepr)
            elif name == "kw_names":
                self.kw_names_op(instruction.arg)
            else:
                getattr(self, name + "_op")(instruction.argval)

            self.next_instr += 1

        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def kw_names_op(self, arg: tp.Any) -> None:
        self.kw = self.code.co_consts[arg]

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        arguments = self.popn(arg)
        f = self.pop()

        if len(self.data_stack) != 0:
            if popa := self.pop():
                arguments = tuple([f] + list(arguments))
                f = popa

        if self.kw is not None:
            kw_len = len(self.kw)
            kw = dict(zip(self.kw, arguments[-kw_len:]))
            arguments = arguments[:kw_len + 1]

            self.push(f(*arguments, **kw))
        else:
            self.push(f(*arguments))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f'Do not have : {arg}')

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f'Do not have : {arg}')

    def delete_global_op(self, arg: str) -> None:
        if arg in self.globals:
            del self.globals[arg]
        elif arg in self.builtins:
            del self.builtins[arg]
        else:
            raise NameError(f'Do not have : {arg}')

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, flags: int) -> None:
        code = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args = bind_args(
                code, *args, **kwargs
            )
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(
                code, self.builtins, self.globals, f_locals
            )  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def binary_op_op(self, op: str) -> None:
        if op[-1] == '=':  # Only top stack elem
            buddy = self.pop()

            if op == '+=':
                self.data_stack[-1] += buddy
            elif op == '-=':
                self.data_stack[-1] -= buddy
            elif op == '*=':
                self.data_stack[-1] *= buddy
            elif op == '/=':
                self.data_stack[-1] /= buddy
            elif op == '//=':
                self.data_stack[-1] //= buddy
            elif op == '**=':
                self.data_stack[-1] **= buddy
            elif op == '%=':
                self.data_stack[-1] %= buddy
            elif op == '<<=':
                self.data_stack[-1] <<= buddy
            elif op == '>>=':
                self.data_stack[-1] >>= buddy
            elif op == '|=':
                self.data_stack[-1] |= buddy
            elif op == '&=':
                self.data_stack[-1] &= buddy
            elif op == '^=':
                self.data_stack[-1] ^= buddy
        else:  # kinda normal operation
            left, right = self.popn(2)

            if op == '+':
                self.push(left + right)
            elif op == '-':
                self.push(left - right)
            elif op == '*':
                self.push(left * right)
            elif op == '/':
                self.push(left / right)
            elif op == '//':
                self.push(left // right)
            elif op == '**':
                self.push(left ** right)
            elif op == '%':
                self.push(left % right)
            elif op == '<<':
                self.push(left << right)
            elif op == '>>':
                self.push(left >> right)
            elif op == '|':
                self.push(left | right)
            elif op == '&':
                self.push(left & right)
            elif op == '^':
                self.push(left ^ right)

    def get_iter_op(self, arg: str) -> None:
        tos = self.pop()
        self.push(iter(tos))

    def my_jump(self, delta: int) -> None:
        instructions = list(dis.get_instructions(self.code))
        for i, instr in enumerate(instructions):
            if instr.offset == delta:
                self.next_instr = i - 1

    def for_iter_op(self, delta: int) -> None:
        try:
            self.push(next(self.top()))
        except StopIteration:
            self.pop()
            self.my_jump(delta)

    def jump_forward_op(self, delta: int) -> None:
        self.my_jump(delta)

    def jump_backward_op(self, delta: int) -> None:
        self.my_jump(delta)

    def unpack_sequence_op(self, count: int) -> None:
        tos = self.pop()

        cut = tos[:count]
        for val in cut[::-1]:
            self.push(val)

    def compare_op_op(self, opname: str) -> None:
        left, right = self.popn(2)

        if opname == '<':
            self.push(left < right)
        elif opname == '<=':
            self.push(left <= right)
        elif opname == '>':
            self.push(left > right)
        elif opname == '>=':
            self.push(left >= right)
        elif opname == '==':
            self.push(left == right)
        elif opname == '!=':
            self.push(left != right)

    def pop_jump_forward_if_false_op(self, delta: int) -> None:
        tos = self.pop()
        if not tos:
            self.my_jump(delta)

    def pop_jump_backward_if_false_op(self, delta: int) -> None:
        tos = self.pop()
        if not tos:
            self.my_jump(delta)

    def pop_jump_forward_if_true_op(self, delta: int) -> None:
        tos = self.pop()
        if tos:
            self.my_jump(delta)

    def pop_jump_backward_if_true_op(self, delta: int) -> None:
        tos = self.pop()
        if tos:
            self.my_jump(delta)

    def pop_jump_forward_if_none_op(self, delta: int) -> None:
        tos = self.pop()
        if tos is None:
            self.my_jump(delta)

    def pop_jump_backward_if_none_op(self, delta: int) -> None:
        tos = self.pop()
        if tos is None:
            self.my_jump(delta)

    def pop_jump_forward_if_not_none_op(self, delta: int) -> None:
        tos = self.pop()
        if tos is not None:
            self.my_jump(delta)

    def pop_jump_backward_if_not_none_op(self, delta: int) -> None:
        tos = self.pop()
        if tos is not None:
            self.my_jump(delta)

    def jump_if_true_or_pop_op(self, delta: int) -> None:
        if self.top():
            self.my_jump(delta)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, delta: int) -> None:
        if not self.top():
            self.my_jump(delta)
        else:
            self.pop()

    def load_assertion_error_op(self, arg: str) -> None:
        self.push(AssertionError)

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            raise
        elif argc == 1:
            raise self.pop()
        elif argc == 2:
            tos1, tos = self.popn(2)
            raise tos1 from tos

    def build_slice_op(self, argc: int) -> None:
        if argc != 2 and argc != 3:
            raise ValueError(f'argc = {argc}')
        if argc == 2:
            tos1, tos = self.popn(2)
            self.push(slice(tos1, tos))
        else:
            tos2, tos1, tos = self.popn(3)
            self.push(slice(tos2, tos1, tos))

    def binary_subscr_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1[tos])

    def build_list_op(self, count: int) -> None:
        items = self.popn(count)
        self.push(items)

    def store_subscr_op(self, arg: str) -> None:
        self.data_stack[-2][self.data_stack[-1]] = self.data_stack[-3]

    def delete_subscr_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]

    def list_extend_op(self, i: int) -> None:
        tos = self.pop()
        list.extend(self.data_stack[-i], tos)

    def build_const_key_map_op(self, count: int) -> None:
        tuple_of_keys = self.pop()
        dct = dict()
        vals = self.popn(count)

        for i, key in enumerate(tuple_of_keys):
            dct[key] = vals[i]
        self.push(dct)

    def build_set_op(self, count: int) -> None:
        items = self.popn(count)
        self.push(set(items))

    def set_update_op(self, i: int) -> None:
        tos = self.pop()
        set.update(self.data_stack[-i], tos)

    def format_value_op(self, flags: tp.Any) -> None:
        func, flag = flags
        value = "" if flag else self.pop()
        value = func(value) if func is not None else str(value)

        self.push(value)

    def build_string_op(self, count: int) -> None:
        strs = self.popn(count)
        self.push(''.join(strs))

    def load_method_op(self, namei: str) -> None:
        tos = self.pop()
        attr = getattr(tos, namei)

        if hasattr(tos, namei) and isinstance(attr, types.MethodType):
            self.push(attr)
            self.push(tos)
        elif hasattr(tos, namei):
            self.push(None)
            self.push(attr)
        else:
            self.push(None)

    def copy_op(self, i: int) -> None:
        self.push(self.data_stack[-i])

    def swap_op(self, i: int) -> None:
        self.data_stack[-i], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-i]

    def is_op_op(self, invert: bool) -> None:
        left, right = self.popn(2)
        if not invert:
            self.push(left is right)
        else:
            self.push(left is not right)

    def contains_op_op(self, invert: bool) -> None:
        left, right = self.popn(2)
        if not invert:
            self.push(left in right)
        else:
            self.push(left not in right)

    def unary_positive_op(self, arg: str) -> None:
        self.data_stack[-1] = +self.data_stack[-1]

    def unary_negative_op(self, arg: str) -> None:
        self.data_stack[-1] = -self.data_stack[-1]

    def unary_invert_op(self, arg: str) -> None:
        self.data_stack[-1] = ~self.data_stack[-1]

    def unary_not_op(self, arg: str) -> None:
        self.data_stack[-1] = not self.data_stack[-1]

    def store_attr_op(self, namei: str) -> None:
        tos1, tos = self.popn(2)
        setattr(tos, namei, tos1)

    def load_attr_op(self, attr: tp.Any) -> None:
        tos = self.pop()
        self.push(getattr(tos, attr))

    def delete_attr_op(self, namei: str) -> None:
        tos = self.pop()
        delattr(tos, namei)

    def delete_name_op(self, namei: str) -> None:
        if namei in self.locals.keys():
            del self.locals[namei]
        elif namei in self.globals.keys():
            del self.globals[namei]
        elif namei in self.builtins.keys():
            del self.builtins[namei]
        else:
            raise NameError(f'Do not have : {namei}')

    def import_name_op(self, namei: str) -> None:
        tos1, tos = self.popn(2)
        self.push(
            __import__(namei, self.globals, self.locals, tos, tos1)
        )

    def import_from_op(self, namei: str) -> None:
        self.push(getattr(self.top(), namei))

    def import_star_op(self, arg: str) -> None:
        tos = self.pop()

        for attr in dir(tos):
            if attr[0] == '_':
                continue
            self.locals[attr] = getattr(tos, attr)

    def list_to_tuple_op(self, arg: str) -> None:
        tos = self.pop()
        self.push(tuple(tos))

    def nop_op(self, arg: str) -> None:
        pass

    def build_map_op(self, count: int) -> None:
        items = self.popn(2 * count)
        ind, map = 0, dict()

        while ind < 2 * count:
            map[items[ind]] = items[ind + 1]
            ind += 2
        self.push(map)

    def dict_update_op(self, i: int) -> None:
        tos = self.pop()
        dict.update(self.data_stack[-i], tos)

    def dict_merge_op(self, i: int) -> None:
        tos = self.pop()
        dict.update(self.data_stack[-i], tos)

    def call_function_ex_op(self, flags: tp.Any) -> None:
        pass

    def build_tuple_op(self, count: int) -> None:
        items = self.popn(count)
        self.push(tuple(items))

    def load_fast_op(self, namei: str) -> None:
        if namei in self.locals:
            self.push(self.locals[namei])
        else:
            raise UnboundLocalError(f'Do not have : {namei}')

    def store_fast_op(self, arg: str) -> None:
        tos = self.pop()
        self.locals[arg] = tos

    def load_build_class_op(self, arg: str) -> None:
        self.push(__build_class__)

    def setup_annotations_op(self, arg: str) -> None:
        if "__annotations__" not in self.locals:
            self.locals["__annotations__"] = dict()


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(
            code_obj,
            builtins.globals()['__builtins__'],
            globals_context,
            globals_context
        )
        return frame.run()
