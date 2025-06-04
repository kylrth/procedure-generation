import argparse
import dataclasses
import sys
from typing import Self


# ruff: noqa: RUF009  # here we define a helper function for adding help text to dataclass fields
# ruff: noqa: A001, A002  # here we're following the dataclasses module


_help_key = "exp.help"
_choices_key = "exp.choices"


def chelp(
    help: str | None = None,
    *,
    choices=None,
    default=dataclasses.MISSING,
    default_factory=dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=dataclasses.MISSING,
):
    if metadata is None:
        metadata = {}
    metadata[_help_key] = help
    if choices is not None:
        metadata[_choices_key] = choices

    return dataclasses.field(  # type: ignore[reportCallIssue]
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


class Config:
    """A base class for defining experiment configuration. Child classes should have the @dataclass
    decorator.

    Fields should use the help function to set help strings. All other arguments are passed to
    dataclasses.field.
    """

    @classmethod
    def __default(cls, f: dataclasses.Field):
        if f.default is not dataclasses.MISSING:
            return f.default
        if hasattr(cls, f.name):
            return getattr(cls, f.name)
        return dataclasses.MISSING

    @staticmethod
    def __flag_name(field_name: str, /, *, negate: bool = False) -> str:
        if negate:
            return "--no-" + field_name
        if len(field_name) == 1:
            return "-" + field_name
        return "--" + field_name.replace("_", "-")

    @classmethod
    def from_args(cls) -> Self:  # noqa: C901, PLR0912
        """Construct a config from command line arguments.

        If child classes need to set defaults depending on values of other fields, they should do so
        in __post_init__ instead of setting a default value.
        """
        parser = argparse.ArgumentParser(description=sys.modules[cls.__module__].__doc__)

        for f in dataclasses.fields(cls):  # type: ignore[reportArgumentType]  # child class will be dataclass
            pargs = {}
            if _help_key in f.metadata:
                pargs["help"] = f.metadata[_help_key]

            default = cls.__default(f)

            if f.type is bool:  # flag
                # a default of None means we'll create both flags
                ok = False
                if default is False or default is None or default is dataclasses.MISSING:
                    parser.add_argument(cls.__flag_name(f.name), action="store_true", **pargs)
                    ok = True
                if default is True or default is None or default is dataclasses.MISSING:
                    parser.add_argument(
                        cls.__flag_name(f.name, negate=True), action="store_true", **pargs
                    )
                    ok = True
                if not ok:
                    raise TypeError("bool field's default is not bool or None or MISSING")
                continue

            if default is dataclasses.MISSING:
                pargs["required"] = True
            else:
                pargs["default"] = default
            if _choices_key in f.metadata:
                pargs["choices"] = f.metadata[_choices_key]
            pargs["type"] = f.type
            parser.add_argument(cls.__flag_name(f.name), **pargs)  # type: ignore[reportArgumentType]

        args = parser.parse_args()

        bool_flags_to_check = []

        init_args = {}
        for f in dataclasses.fields(cls):  # type: ignore[reportArgumentType]  # child class will be dataclass
            if f.type is bool:  # flag
                default = cls.__default(f)
                if default is dataclasses.MISSING:
                    default = None

                # None or MISSING
                set_pos = getattr(args, f.name, False)
                set_neg = getattr(args, f"no_{f.name}", False)
                if set_pos and set_neg:
                    parser.error(
                        f"cannot set both {cls.__flag_name(f.name)} and "
                        f"{cls.__flag_name(f.name, negate=True)}"
                    )
                if set_pos:
                    init_args[f.name] = True
                    continue
                if set_neg:
                    init_args[f.name] = False
                    continue
                # no flags set
                # if default is None, will expect child class to set in __post_init__
                init_args[f.name] = default
                if default is not True and default is not False:
                    bool_flags_to_check.append(f.name)
                continue

            init_args[f.name] = getattr(args, f.name)

        out = cls(**init_args)

        for name in bool_flags_to_check:
            if getattr(out, name) is None:
                raise TypeError(f"{cls}.__post_init__ did not handle unset default bool")

        return out
