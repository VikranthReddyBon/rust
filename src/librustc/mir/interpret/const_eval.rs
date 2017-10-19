use ty::{self, TyCtxt, Ty, Instance, layout};
use mir;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use super::{EvalResult, EvalError, EvalErrorKind, GlobalId, Lvalue, Value, PrimVal, EvalContext,
            StackPopCleanup, PtrAndAlign, ValTy, HasMemory};

use rustc_const_math::ConstInt;

use std::fmt;
use std::error::Error;

pub fn eval_body<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, (PtrAndAlign, Ty<'tcx>)> {
    debug!("eval_body: {:?}, {:?}", instance, param_env);
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::<CompileTimeFunctionEvaluator>::new(tcx, limits, param_env, ());
    let cid = GlobalId {
        instance,
        promoted: None,
    };
    if ecx.tcx.has_attr(instance.def_id(), "linkage") {
        return Err(ConstEvalError::NotConst("extern global".to_string()).into());
    }

    let mir = ecx.load_mir(instance.def)?;
    if tcx.interpret_interner.borrow().get_cached(cid).is_none() {
        let size = ecx.type_size_with_substs(mir.return_ty, instance.substs)?
            .expect("unsized global");
        let align = ecx.type_align_with_substs(mir.return_ty, instance.substs)?;
        let ptr = ecx.memory.allocate(
            size,
            align,
            None,
        )?;
        let aligned = !ecx.is_packed_with_substs(mir.return_ty, instance.substs)?;
        tcx.interpret_interner.borrow_mut().cache(
            cid,
            PtrAndAlign {
                ptr: ptr.into(),
                aligned,
            },
        );
        let cleanup = StackPopCleanup::MarkStatic(Mutability::Immutable);
        let name = ty::tls::with(|tcx| tcx.item_path_str(instance.def_id()));
        trace!("const_eval: pushing stack frame for global: {}", name);
        ecx.push_stack_frame(
            instance,
            mir.span,
            mir,
            Lvalue::from_ptr(ptr),
            cleanup,
        )?;

        while ecx.step()? {}
    }
    let value = tcx.interpret_interner.borrow().get_cached(cid).expect("global not cached");
    let ret_ty = ecx.monomorphize(mir.return_ty, instance.substs);
    Ok((value, ret_ty))
}

pub fn eval_body_as_integer<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: Instance<'tcx>,
) -> EvalResult<'tcx, ConstInt> {
    let (ptr, ty) = eval_body(tcx, instance, param_env)?;
    let limits = super::ResourceLimits::default();
    let ecx = EvalContext::<CompileTimeFunctionEvaluator>::new(tcx, limits, param_env, ());
    let prim = match ecx.read_maybe_aligned(ptr.aligned, |ectx| ectx.try_read_value(ptr.ptr, ty))? {
        Some(Value::ByVal(prim)) => prim.to_bytes()?,
        _ => return err!(TypeNotPrimitive(ty)),
    };
    use syntax::ast::{IntTy, UintTy};
    use ty::TypeVariants::*;
    use rustc_const_math::{ConstIsize, ConstUsize};
    Ok(match ty.sty {
        TyInt(IntTy::I8) => ConstInt::I8(prim as i128 as i8),
        TyInt(IntTy::I16) => ConstInt::I16(prim as i128 as i16),
        TyInt(IntTy::I32) => ConstInt::I32(prim as i128 as i32),
        TyInt(IntTy::I64) => ConstInt::I64(prim as i128 as i64),
        TyInt(IntTy::I128) => ConstInt::I128(prim as i128),
        TyInt(IntTy::Is) => ConstInt::Isize(
            ConstIsize::new(prim as i128 as i64, tcx.sess.target.isize_ty)
                .expect("miri should already have errored"),
        ),
        TyUint(UintTy::U8) => ConstInt::U8(prim as u8),
        TyUint(UintTy::U16) => ConstInt::U16(prim as u16),
        TyUint(UintTy::U32) => ConstInt::U32(prim as u32),
        TyUint(UintTy::U64) => ConstInt::U64(prim as u64),
        TyUint(UintTy::U128) => ConstInt::U128(prim),
        TyUint(UintTy::Us) => ConstInt::Usize(
            ConstUsize::new(prim as u64, tcx.sess.target.usize_ty)
                .expect("miri should already have errored"),
        ),
        _ => {
            return Err(
                ConstEvalError::NeedsRfc(
                    "evaluating anything other than isize/usize during typeck".to_string(),
                ).into(),
            )
        }
    })
}

pub struct CompileTimeFunctionEvaluator;

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalErrorKind::MachineError(Box::new(self)).into()
    }
}

#[derive(Clone, Debug)]
enum ConstEvalError {
    NeedsRfc(String),
    NotConst(String),
}

impl fmt::Display for ConstEvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(
                    f,
                    "\"{}\" needs an rfc before being allowed inside constants",
                    msg
                )
            }
            NotConst(ref msg) => write!(f, "Cannot evaluate within constants: \"{}\"", msg),
        }
    }
}

impl Error for ConstEvalError {
    fn description(&self) -> &str {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(_) => "this feature needs an rfc before being allowed inside constants",
            NotConst(_) => "this feature is not compatible with constant evaluation",
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl<'tcx> super::Machine<'tcx> for CompileTimeFunctionEvaluator {
    type Data = ty::ParamEnv<'tcx>;
    type MemoryData = ();
    type MemoryKinds = !;
    fn param_env<'a>(
        ecx: &EvalContext<'a, 'tcx, Self>,
    ) -> ty::ParamEnv<'tcx> {
        ecx.machine_data
    }
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        _args: &[ValTy<'tcx>],
        span: Span,
        _sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        debug!("eval_fn_call: {:?}", instance);
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            return Err(
                ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into(),
            );
        }
        let mir = match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                // some simple things like `malloc` might get accepted in the future
                return Err(
                    ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                        .into(),
                );
            }
            Err(other) => return Err(other),
        };
        let (return_lvalue, return_to_block) = match destination {
            Some((lvalue, block)) => (lvalue, StackPopCleanup::Goto(block)),
            None => (Lvalue::undef(), StackPopCleanup::None),
        };

        ecx.push_stack_frame(
            instance,
            span,
            mir,
            return_lvalue,
            return_to_block,
        )?;

        Ok(false)
    }


    fn call_intrinsic<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        _args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: Ty<'tcx>,
        _dest_layout: &'tcx layout::Layout,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &ecx.tcx.item_name(instance.def_id())[..];
        match intrinsic_name {
            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = ecx.type_align(elem_ty)?;
                let align_val = PrimVal::from_u128(elem_align as u128);
                ecx.write_primval(dest, align_val, dest_ty)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = ecx.type_size(ty)?.expect(
                    "size_of intrinsic called on unsized value",
                ) as u128;
                ecx.write_primval(dest, PrimVal::from_u128(size), dest_ty)?;
            }

            name => return Err(ConstEvalError::NeedsRfc(format!("calling intrinsic `{}`", name)).into()),
        }

        ecx.goto_block(target);

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn try_ptr_op<'a>(
        _ecx: &EvalContext<'a, 'tcx, Self>,
        _bin_op: mir::BinOp,
        left: PrimVal,
        _left_ty: Ty<'tcx>,
        right: PrimVal,
        _right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        if left.is_bytes() && right.is_bytes() {
            Ok(None)
        } else {
            Err(
                ConstEvalError::NeedsRfc("Pointer arithmetic or comparison".to_string()).into(),
            )
        }
    }

    fn mark_static_initialized(m: !) -> EvalResult<'tcx> {
        m
    }

    fn box_alloc<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _ty: ty::Ty<'tcx>,
        _dest: Lvalue,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("Heap allocations via `box` keyword".to_string()).into(),
        )
    }

    fn global_item_with_linkage<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _mutability: Mutability,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NotConst("statics with `linkage` attribute".to_string()).into(),
        )
    }
}