// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::const_val::ConstVal::*;
use rustc::middle::const_val::ConstAggregate::*;
use rustc::middle::const_val::ErrKind::*;
use rustc::middle::const_val::{ByteArray, ConstVal, ConstEvalErr, EvalResult, ErrKind};

use rustc::hir::map as hir_map;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::traits;
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::ty::util::IntTypeExt;
use rustc::ty::subst::{Substs, Subst};
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::NodeMap;

use rustc::mir::interpret::{PrimVal, Value, PtrAndAlign, HasMemory, EvalError};
use rustc::mir::Field;
use rustc::mir::interpret::{Lvalue, LvalueExtra};
use rustc_data_structures::indexed_vec::Idx;

use syntax::abi::Abi;
use syntax::ast;
use syntax::attr;
use rustc::hir::{self, Expr};
use syntax_pos::Span;

use std::cmp::Ordering;

use rustc_const_math::*;
macro_rules! signal {
    ($e:expr, $exn:expr) => {
        return Err(ConstEvalErr { span: $e.span, kind: $exn })
    }
}

macro_rules! math {
    ($e:expr, $op:expr) => {
        match $op {
            Ok(val) => val,
            Err(e) => signal!($e, ErrKind::from(e)),
        }
    }
}

/// * `DefId` is the id of the constant.
/// * `Substs` is the monomorphized substitutions for the expression.
pub fn lookup_const_by_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                                    -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let (def_id, _) = key.value;
    if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
        match tcx.hir.find(node_id) {
            Some(hir_map::NodeTraitItem(_)) => {
                // If we have a trait item and the substitutions for it,
                // `resolve_trait_associated_const` will select an impl
                // or the default.
                resolve_trait_associated_const(tcx, key)
            }
            _ => Some(key.value)
        }
    } else {
        match tcx.describe_def(def_id) {
            Some(Def::AssociatedConst(_)) => {
                // As mentioned in the comments above for in-crate
                // constants, we only try to find the expression for a
                // trait-associated const if the caller gives us the
                // substitutions for the reference to it.
                if tcx.trait_of_item(def_id).is_some() {
                    resolve_trait_associated_const(tcx, key)
                } else {
                    Some(key.value)
                }
            }
            _ => Some(key.value)
        }
    }
}

pub struct ConstContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    substs: &'tcx Substs<'tcx>,
    fn_args: Option<NodeMap<&'tcx ty::Const<'tcx>>>
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               param_env_and_substs: ty::ParamEnvAnd<'tcx, &'tcx Substs<'tcx>>,
               tables: &'a ty::TypeckTables<'tcx>)
               -> Self {
        ConstContext {
            tcx,
            param_env: param_env_and_substs.param_env,
            tables,
            substs: param_env_and_substs.value,
            fn_args: None
        }
    }

    /// Evaluate a constant expression in a context where the expression isn't
    /// guaranteed to be evaluable.
    pub fn eval(&self, e: &'tcx Expr) -> EvalResult<'tcx> {
        if self.tables.tainted_by_errors {
            signal!(e, TypeckError);
        }
        eval_const_expr_partial(self, e)
    }
}

type CastResult<'tcx> = Result<ConstVal<'tcx>, ErrKind<'tcx>>;

fn eval_const_expr_partial<'a, 'tcx>(cx: &ConstContext<'a, 'tcx>,
                                     e: &'tcx Expr) -> EvalResult<'tcx> {
    let tcx = cx.tcx;
    let ty = cx.tables.expr_ty(e).subst(tcx, cx.substs);
    let mk_const = |val| tcx.mk_const(ty::Const { val, ty });

    let result = match e.node {
      hir::ExprUnary(hir::UnNeg, ref inner) => {
        // unary neg literals already got their sign during creation
        if let hir::ExprLit(ref lit) = inner.node {
            use syntax::ast::*;
            use syntax::ast::LitIntType::*;
            const I8_OVERFLOW: u128 = i8::min_value() as u8 as u128;
            const I16_OVERFLOW: u128 = i16::min_value() as u16 as u128;
            const I32_OVERFLOW: u128 = i32::min_value() as u32 as u128;
            const I64_OVERFLOW: u128 = i64::min_value() as u64 as u128;
            const I128_OVERFLOW: u128 = i128::min_value() as u128;
            let negated = match (&lit.node, &ty.sty) {
                (&LitKind::Int(I8_OVERFLOW, _), &ty::TyInt(IntTy::I8)) |
                (&LitKind::Int(I8_OVERFLOW, Signed(IntTy::I8)), _) => {
                    Some(I8(i8::min_value()))
                },
                (&LitKind::Int(I16_OVERFLOW, _), &ty::TyInt(IntTy::I16)) |
                (&LitKind::Int(I16_OVERFLOW, Signed(IntTy::I16)), _) => {
                    Some(I16(i16::min_value()))
                },
                (&LitKind::Int(I32_OVERFLOW, _), &ty::TyInt(IntTy::I32)) |
                (&LitKind::Int(I32_OVERFLOW, Signed(IntTy::I32)), _) => {
                    Some(I32(i32::min_value()))
                },
                (&LitKind::Int(I64_OVERFLOW, _), &ty::TyInt(IntTy::I64)) |
                (&LitKind::Int(I64_OVERFLOW, Signed(IntTy::I64)), _) => {
                    Some(I64(i64::min_value()))
                },
                (&LitKind::Int(I128_OVERFLOW, _), &ty::TyInt(IntTy::I128)) |
                (&LitKind::Int(I128_OVERFLOW, Signed(IntTy::I128)), _) => {
                    Some(I128(i128::min_value()))
                },
                (&LitKind::Int(n, _), &ty::TyInt(IntTy::Is)) |
                (&LitKind::Int(n, Signed(IntTy::Is)), _) => {
                    match tcx.sess.target.isize_ty {
                        IntTy::I16 => if n == I16_OVERFLOW {
                            Some(Isize(Is16(i16::min_value())))
                        } else {
                            None
                        },
                        IntTy::I32 => if n == I32_OVERFLOW {
                            Some(Isize(Is32(i32::min_value())))
                        } else {
                            None
                        },
                        IntTy::I64 => if n == I64_OVERFLOW {
                            Some(Isize(Is64(i64::min_value())))
                        } else {
                            None
                        },
                        _ => span_bug!(e.span, "typeck error")
                    }
                },
                _ => None
            };
            if let Some(i) = negated {
                return Ok(mk_const(Integral(i)));
            }
        }
        mk_const(match cx.eval(inner)?.val {
          Float(f) => Float(-f),
          Integral(i) => Integral(math!(e, -i)),
          _ => signal!(e, TypeckError)
        })
      }
      hir::ExprUnary(hir::UnNot, ref inner) => {
        mk_const(match cx.eval(inner)?.val {
          Integral(i) => Integral(math!(e, !i)),
          Bool(b) => Bool(!b),
          _ => signal!(e, TypeckError)
        })
      }
      hir::ExprUnary(hir::UnDeref, _) => signal!(e, UnimplementedConstVal("deref operation")),
      hir::ExprBinary(op, ref a, ref b) => {
        // technically, if we don't have type hints, but integral eval
        // gives us a type through a type-suffix, cast or const def type
        // we need to re-eval the other value of the BinOp if it was
        // not inferred
        mk_const(match (cx.eval(a)?.val, cx.eval(b)?.val) {
          (Float(a), Float(b)) => {
            use std::cmp::Ordering::*;
            match op.node {
              hir::BiAdd => Float(math!(e, a + b)),
              hir::BiSub => Float(math!(e, a - b)),
              hir::BiMul => Float(math!(e, a * b)),
              hir::BiDiv => Float(math!(e, a / b)),
              hir::BiRem => Float(math!(e, a % b)),
              hir::BiEq => Bool(math!(e, a.try_cmp(b)) == Equal),
              hir::BiLt => Bool(math!(e, a.try_cmp(b)) == Less),
              hir::BiLe => Bool(math!(e, a.try_cmp(b)) != Greater),
              hir::BiNe => Bool(math!(e, a.try_cmp(b)) != Equal),
              hir::BiGe => Bool(math!(e, a.try_cmp(b)) != Less),
              hir::BiGt => Bool(math!(e, a.try_cmp(b)) == Greater),
              _ => span_bug!(e.span, "typeck error"),
            }
          }
          (Integral(a), Integral(b)) => {
            use std::cmp::Ordering::*;
            match op.node {
              hir::BiAdd => Integral(math!(e, a + b)),
              hir::BiSub => Integral(math!(e, a - b)),
              hir::BiMul => Integral(math!(e, a * b)),
              hir::BiDiv => Integral(math!(e, a / b)),
              hir::BiRem => Integral(math!(e, a % b)),
              hir::BiBitAnd => Integral(math!(e, a & b)),
              hir::BiBitOr => Integral(math!(e, a | b)),
              hir::BiBitXor => Integral(math!(e, a ^ b)),
              hir::BiShl => Integral(math!(e, a << b)),
              hir::BiShr => Integral(math!(e, a >> b)),
              hir::BiEq => Bool(math!(e, a.try_cmp(b)) == Equal),
              hir::BiLt => Bool(math!(e, a.try_cmp(b)) == Less),
              hir::BiLe => Bool(math!(e, a.try_cmp(b)) != Greater),
              hir::BiNe => Bool(math!(e, a.try_cmp(b)) != Equal),
              hir::BiGe => Bool(math!(e, a.try_cmp(b)) != Less),
              hir::BiGt => Bool(math!(e, a.try_cmp(b)) == Greater),
              _ => span_bug!(e.span, "typeck error"),
            }
          }
          (Bool(a), Bool(b)) => {
            Bool(match op.node {
              hir::BiAnd => a && b,
              hir::BiOr => a || b,
              hir::BiBitXor => a ^ b,
              hir::BiBitAnd => a & b,
              hir::BiBitOr => a | b,
              hir::BiEq => a == b,
              hir::BiNe => a != b,
              hir::BiLt => a < b,
              hir::BiLe => a <= b,
              hir::BiGe => a >= b,
              hir::BiGt => a > b,
              _ => span_bug!(e.span, "typeck error"),
             })
          }
          (Char(a), Char(b)) => {
            Bool(match op.node {
              hir::BiEq => a == b,
              hir::BiNe => a != b,
              hir::BiLt => a < b,
              hir::BiLe => a <= b,
              hir::BiGe => a >= b,
              hir::BiGt => a > b,
              _ => span_bug!(e.span, "typeck error"),
             })
          }

          _ => signal!(e, MiscBinaryOp),
        })
      }
      hir::ExprCast(ref base, _) => {
        let base_val = cx.eval(base)?;
        let base_ty = cx.tables.expr_ty(base).subst(tcx, cx.substs);
        if ty == base_ty {
            base_val
        } else {
            match cast_const(tcx, base_val.val, ty) {
                Ok(val) => mk_const(val),
                Err(kind) => signal!(e, kind),
            }
        }
      }
      hir::ExprPath(ref qpath) => {
        let substs = cx.tables.node_substs(e.hir_id).subst(tcx, cx.substs);
          match cx.tables.qpath_def(qpath, e.hir_id) {
              Def::Const(def_id) |
              Def::AssociatedConst(def_id) => {
                    match tcx.at(e.span).const_eval(cx.param_env.and((def_id, substs))) {
                        Ok(val) => val,
                        Err(ConstEvalErr { kind: TypeckError, .. }) => {
                            signal!(e, TypeckError);
                        }
                        Err(err) => {
                            debug!("bad reference: {:?}, {:?}", err.description(), err.span);
                            signal!(e, ErroneousReferencedConstant(box err))
                        },
                    }
              },
              Def::VariantCtor(variant_def, CtorKind::Const) => {
                mk_const(Variant(variant_def))
              }
              Def::VariantCtor(_, CtorKind::Fn) => {
                  signal!(e, UnimplementedConstVal("enum variants"));
              }
              Def::StructCtor(_, CtorKind::Const) => {
                  mk_const(Aggregate(Struct(&[])))
              }
              Def::StructCtor(_, CtorKind::Fn) => {
                  signal!(e, UnimplementedConstVal("tuple struct constructors"))
              }
              Def::Local(id) => {
                  debug!("Def::Local({:?}): {:?}", id, cx.fn_args);
                  if let Some(&val) = cx.fn_args.as_ref().and_then(|args| args.get(&id)) {
                      val
                  } else {
                      signal!(e, NonConstPath);
                  }
              },
              Def::Method(id) | Def::Fn(id) => mk_const(Function(id, substs)),
              Def::Err => span_bug!(e.span, "typeck error"),
              _ => signal!(e, NonConstPath),
          }
      }
      hir::ExprCall(ref callee, ref args) => {
          let (def_id, substs) = match cx.eval(callee)?.val {
              Function(def_id, substs) => (def_id, substs),
              _ => signal!(e, TypeckError),
          };

          if tcx.fn_sig(def_id).abi() == Abi::RustIntrinsic {
            let layout_of = |ty: Ty<'tcx>| {
                let ty = tcx.erase_regions(&ty);
                tcx.at(e.span).layout_raw(cx.param_env.reveal_all().and(ty)).map_err(|err| {
                    ConstEvalErr { span: e.span, kind: LayoutError(err) }
                })
            };
            match &tcx.item_name(def_id)[..] {
                "size_of" => {
                    let size = layout_of(substs.type_at(0))?.size(tcx).bytes();
                    return Ok(mk_const(Integral(Usize(ConstUsize::new(size,
                        tcx.sess.target.usize_ty).unwrap()))));
                }
                "min_align_of" => {
                    let align = layout_of(substs.type_at(0))?.align(tcx).abi();
                    return Ok(mk_const(Integral(Usize(ConstUsize::new(align,
                        tcx.sess.target.usize_ty).unwrap()))));
                }
                _ => signal!(e, TypeckError)
            }
          }

          let body = if let Some(node_id) = tcx.hir.as_local_node_id(def_id) {
            if let Some(fn_like) = FnLikeNode::from_node(tcx.hir.get(node_id)) {
                if fn_like.constness() == hir::Constness::Const {
                    tcx.hir.body(fn_like.body())
                } else {
                    signal!(e, TypeckError)
                }
            } else {
                signal!(e, TypeckError)
            }
          } else {
            if tcx.is_const_fn(def_id) {
                tcx.extern_const_body(def_id).body
            } else {
                signal!(e, TypeckError)
            }
          };

          let arg_ids = body.arguments.iter().map(|arg| match arg.pat.node {
               hir::PatKind::Binding(_, canonical_id, _, _) => Some(canonical_id),
               _ => None
           }).collect::<Vec<_>>();
          assert_eq!(arg_ids.len(), args.len());

          let mut call_args = NodeMap();
          for (arg, arg_expr) in arg_ids.into_iter().zip(args.iter()) {
              let arg_val = cx.eval(arg_expr)?;
              debug!("const call arg: {:?}", arg);
              if let Some(id) = arg {
                assert!(call_args.insert(id, arg_val).is_none());
              }
          }
          debug!("const call({:?})", call_args);
          let callee_cx = ConstContext {
            tcx,
            param_env: cx.param_env,
            tables: tcx.typeck_tables_of(def_id),
            substs,
            fn_args: Some(call_args)
          };
          callee_cx.eval(&body.value)?
      },
      hir::ExprLit(ref lit) => match lit_to_const(&lit.node, tcx, ty) {
          Ok(val) => mk_const(val),
          Err(err) => signal!(e, err),
      },
      hir::ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => cx.eval(expr)?,
            None => mk_const(Aggregate(Tuple(&[]))),
        }
      }
      hir::ExprType(ref e, _) => cx.eval(e)?,
      hir::ExprTup(ref fields) => {
        let values = fields.iter().map(|e| cx.eval(e)).collect::<Result<Vec<_>, _>>()?;
        mk_const(Aggregate(Tuple(tcx.alloc_const_slice(&values))))
      }
      hir::ExprStruct(_, ref fields, _) => {
        mk_const(Aggregate(Struct(tcx.alloc_name_const_slice(&fields.iter().map(|f| {
            cx.eval(&f.expr).map(|v| (f.name.node, v))
        }).collect::<Result<Vec<_>, _>>()?))))
      }
      hir::ExprIndex(ref arr, ref idx) => {
        if !tcx.sess.features.borrow().const_indexing {
            signal!(e, IndexOpFeatureGated);
        }
        let arr = cx.eval(arr)?;
        let idx = match cx.eval(idx)?.val {
            Integral(Usize(i)) => i.as_u64(),
            _ => signal!(idx, IndexNotUsize),
        };
        assert_eq!(idx as usize as u64, idx);
        match arr.val {
            Aggregate(Array(v)) => {
                if let Some(&elem) = v.get(idx as usize) {
                    elem
                } else {
                    let n = v.len() as u64;
                    signal!(e, IndexOutOfBounds { len: n, index: idx })
                }
            }

            Aggregate(Repeat(.., n)) if idx >= n => {
                signal!(e, IndexOutOfBounds { len: n, index: idx })
            }
            Aggregate(Repeat(elem, _)) => elem,

            ByteStr(b) if idx >= b.data.len() as u64 => {
                signal!(e, IndexOutOfBounds { len: b.data.len() as u64, index: idx })
            }
            ByteStr(b) => {
                mk_const(Integral(U8(b.data[idx as usize])))
            },

            _ => signal!(e, IndexedNonVec),
        }
      }
      hir::ExprArray(ref v) => {
        let values = v.iter().map(|e| cx.eval(e)).collect::<Result<Vec<_>, _>>()?;
        mk_const(Aggregate(Array(tcx.alloc_const_slice(&values))))
      }
      hir::ExprRepeat(ref elem, _) => {
          let n = match ty.sty {
            ty::TyArray(_, n) => n.val.to_const_int().unwrap().to_u64().unwrap(),
            _ => span_bug!(e.span, "typeck error")
          };
          mk_const(Aggregate(Repeat(cx.eval(elem)?, n)))
      },
      hir::ExprTupField(ref base, index) => {
        if let Aggregate(Tuple(fields)) = cx.eval(base)?.val {
            fields[index.node]
        } else {
            signal!(base, ExpectedConstTuple);
        }
      }
      hir::ExprField(ref base, field_name) => {
        if let Aggregate(Struct(fields)) = cx.eval(base)?.val {
            if let Some(&(_, f)) = fields.iter().find(|&&(name, _)| name == field_name.node) {
                f
            } else {
                signal!(e, MissingStructField);
            }
        } else {
            signal!(base, ExpectedConstStruct);
        }
      }
      hir::ExprAddrOf(..) => signal!(e, UnimplementedConstVal("address operator")),
      _ => signal!(e, MiscCatchAll)
    };

    Ok(result)
}

fn resolve_trait_associated_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                            key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                                            -> Option<(DefId, &'tcx Substs<'tcx>)> {
    let param_env = key.param_env;
    let (def_id, substs) = key.value;
    let trait_item = tcx.associated_item(def_id);
    let trait_id = trait_item.container.id();
    let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, substs));
    debug!("resolve_trait_associated_const: trait_ref={:?}",
           trait_ref);

    tcx.infer_ctxt().enter(|infcx| {
        let mut selcx = traits::SelectionContext::new(&infcx);
        let obligation = traits::Obligation::new(traits::ObligationCause::dummy(),
                                                 param_env,
                                                 trait_ref.to_poly_trait_predicate());
        let selection = match selcx.select(&obligation) {
            Ok(Some(vtable)) => vtable,
            // Still ambiguous, so give up and let the caller decide whether this
            // expression is really needed yet. Some associated constant values
            // can't be evaluated until monomorphization is done in trans.
            Ok(None) => {
                return None
            }
            Err(_) => {
                return None
            }
        };

        // NOTE: this code does not currently account for specialization, but when
        // it does so, it should hook into the param_env.reveal to determine when the
        // constant should resolve.
        match selection {
            traits::VtableImpl(ref impl_data) => {
                let name = trait_item.name;
                let ac = tcx.associated_items(impl_data.impl_def_id)
                    .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
                match ac {
                    // FIXME(eddyb) Use proper Instance resolution to
                    // get the correct Substs returned from here.
                    Some(ic) => {
                        let substs = Substs::identity_for_item(tcx, ic.def_id);
                        Some((ic.def_id, substs))
                    }
                    None => {
                        if trait_item.defaultness.has_value() {
                            Some(key.value)
                        } else {
                            None
                        }
                    }
                }
            }
            traits::VtableParam(_) => None,
            _ => {
                bug!("resolve_trait_associated_const: unexpected vtable type {:?}", selection)
            }
        }
    })
}

fn cast_const_int<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            val: ConstInt,
                            ty: Ty<'tcx>)
                            -> CastResult<'tcx> {
    let v = val.to_u128_unchecked();
    match ty.sty {
        ty::TyBool if v == 0 => Ok(Bool(false)),
        ty::TyBool if v == 1 => Ok(Bool(true)),
        ty::TyInt(ast::IntTy::I8) => Ok(Integral(I8(v as i128 as i8))),
        ty::TyInt(ast::IntTy::I16) => Ok(Integral(I16(v as i128 as i16))),
        ty::TyInt(ast::IntTy::I32) => Ok(Integral(I32(v as i128 as i32))),
        ty::TyInt(ast::IntTy::I64) => Ok(Integral(I64(v as i128 as i64))),
        ty::TyInt(ast::IntTy::I128) => Ok(Integral(I128(v as i128))),
        ty::TyInt(ast::IntTy::Is) => {
            Ok(Integral(Isize(ConstIsize::new_truncating(v as i128, tcx.sess.target.isize_ty))))
        },
        ty::TyUint(ast::UintTy::U8) => Ok(Integral(U8(v as u8))),
        ty::TyUint(ast::UintTy::U16) => Ok(Integral(U16(v as u16))),
        ty::TyUint(ast::UintTy::U32) => Ok(Integral(U32(v as u32))),
        ty::TyUint(ast::UintTy::U64) => Ok(Integral(U64(v as u64))),
        ty::TyUint(ast::UintTy::U128) => Ok(Integral(U128(v as u128))),
        ty::TyUint(ast::UintTy::Us) => {
            Ok(Integral(Usize(ConstUsize::new_truncating(v, tcx.sess.target.usize_ty))))
        },
        ty::TyFloat(fty) => {
            if let Some(i) = val.to_u128() {
                Ok(Float(ConstFloat::from_u128(i, fty)))
            } else {
                // The value must be negative, go through signed integers.
                let i = val.to_u128_unchecked() as i128;
                Ok(Float(ConstFloat::from_i128(i, fty)))
            }
        }
        ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting an address to a raw ptr")),
        ty::TyChar => match val {
            U8(u) => Ok(Char(u as char)),
            _ => bug!(),
        },
        _ => Err(CannotCast),
    }
}

fn cast_const_float<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              val: ConstFloat,
                              ty: Ty<'tcx>) -> CastResult<'tcx> {
    let int_width = |ty| {
        ty::layout::Integer::from_attr(tcx, ty).size().bits() as usize
    };
    match ty.sty {
        ty::TyInt(ity) => {
            if let Some(i) = val.to_i128(int_width(attr::SignedInt(ity))) {
                cast_const_int(tcx, I128(i), ty)
            } else {
                Err(CannotCast)
            }
        }
        ty::TyUint(uty) => {
            if let Some(i) = val.to_u128(int_width(attr::UnsignedInt(uty))) {
                cast_const_int(tcx, U128(i), ty)
            } else {
                Err(CannotCast)
            }
        }
        ty::TyFloat(fty) => Ok(Float(val.convert(fty))),
        _ => Err(CannotCast),
    }
}

fn cast_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        val: ConstVal<'tcx>,
                        ty: Ty<'tcx>)
                        -> CastResult<'tcx> {
    match val {
        Integral(i) => cast_const_int(tcx, i, ty),
        Bool(b) => cast_const_int(tcx, U8(b as u8), ty),
        Float(f) => cast_const_float(tcx, f, ty),
        Char(c) => cast_const_int(tcx, U32(c as u32), ty),
        Variant(v) => {
            let adt = tcx.adt_def(tcx.parent_def_id(v).unwrap());
            let idx = adt.variant_index_with_id(v);
            cast_const_int(tcx, adt.discriminant_for_variant(tcx, idx), ty)
        }
        Function(..) => Err(UnimplementedConstVal("casting fn pointers")),
        ByteStr(b) => match ty.sty {
            ty::TyRawPtr(_) => {
                Err(ErrKind::UnimplementedConstVal("casting a bytestr to a raw ptr"))
            },
            ty::TyRef(_, ty::TypeAndMut { ref ty, mutbl: hir::MutImmutable }) => match ty.sty {
                ty::TyArray(ty, n) => {
                    let n = n.val.to_const_int().unwrap().to_u64().unwrap();
                    if ty == tcx.types.u8 && n == b.data.len() as u64 {
                        Ok(val)
                    } else {
                        Err(CannotCast)
                    }
                }
                ty::TySlice(_) => {
                    Err(ErrKind::UnimplementedConstVal("casting a bytestr to slice"))
                },
                _ => Err(CannotCast),
            },
            _ => Err(CannotCast),
        },
        Str(s) => match ty.sty {
            ty::TyRawPtr(_) => Err(ErrKind::UnimplementedConstVal("casting a str to a raw ptr")),
            ty::TyRef(_, ty::TypeAndMut { ref ty, mutbl: hir::MutImmutable }) => match ty.sty {
                ty::TyStr => Ok(Str(s)),
                _ => Err(CannotCast),
            },
            _ => Err(CannotCast),
        },
        _ => Err(CannotCast),
    }
}

fn lit_to_const<'a, 'tcx>(lit: &'tcx ast::LitKind,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          mut ty: Ty<'tcx>)
                          -> Result<ConstVal<'tcx>, ErrKind<'tcx>> {
    use syntax::ast::*;
    use syntax::ast::LitIntType::*;

    if let ty::TyAdt(adt, _) = ty.sty {
        if adt.is_enum() {
            ty = adt.repr.discr_type().to_ty(tcx)
        }
    }

    match *lit {
        LitKind::Str(ref s, _) => Ok(Str(s.as_str())),
        LitKind::ByteStr(ref data) => Ok(ByteStr(ByteArray { data })),
        LitKind::Byte(n) => Ok(Integral(U8(n))),
        LitKind::Int(n, hint) => {
            match (&ty.sty, hint) {
                (&ty::TyInt(ity), _) |
                (_, Signed(ity)) => {
                    Ok(Integral(ConstInt::new_signed_truncating(n as i128,
                        ity, tcx.sess.target.isize_ty)))
                }
                (&ty::TyUint(uty), _) |
                (_, Unsigned(uty)) => {
                    Ok(Integral(ConstInt::new_unsigned_truncating(n as u128,
                        uty, tcx.sess.target.usize_ty)))
                }
                _ => bug!()
            }
        }
        LitKind::Float(n, fty) => {
            parse_float(&n.as_str(), fty).map(Float)
        }
        LitKind::FloatUnsuffixed(n) => {
            let fty = match ty.sty {
                ty::TyFloat(fty) => fty,
                _ => bug!()
            };
            parse_float(&n.as_str(), fty).map(Float)
        }
        LitKind::Bool(b) => Ok(Bool(b)),
        LitKind::Char(c) => Ok(Char(c)),
    }
}

fn parse_float<'tcx>(num: &str, fty: ast::FloatTy)
                     -> Result<ConstFloat, ErrKind<'tcx>> {
    ConstFloat::from_str(num, fty).map_err(|_| {
        // FIXME(#31407) this is only necessary because float parsing is buggy
        UnimplementedConstVal("could not evaluate float literal (see issue #31407)")
    })
}

pub fn compare_const_vals(tcx: TyCtxt, span: Span, a: &ConstVal, b: &ConstVal)
                          -> Result<Ordering, ErrorReported>
{
    let result = match (a, b) {
        (&Integral(a), &Integral(b)) => a.try_cmp(b).ok(),
        (&Float(a), &Float(b)) => a.try_cmp(b).ok(),
        (&Str(ref a), &Str(ref b)) => Some(a.cmp(b)),
        (&Bool(a), &Bool(b)) => Some(a.cmp(&b)),
        (&ByteStr(a), &ByteStr(b)) => Some(a.data.cmp(b.data)),
        (&Char(a), &Char(b)) => Some(a.cmp(&b)),
        _ => None,
    };

    match result {
        Some(result) => Ok(result),
        None => {
            // FIXME: can this ever be reached?
            tcx.sess.delay_span_bug(span,
                &format!("type mismatch comparing {:?} and {:?}", a, b));
            Err(ErrorReported)
        }
    }
}

impl<'a, 'tcx> ConstContext<'a, 'tcx> {
    pub fn compare_lit_exprs(&self,
                             span: Span,
                             a: &'tcx Expr,
                             b: &'tcx Expr) -> Result<Ordering, ErrorReported> {
        let tcx = self.tcx;
        let a = match self.eval(a) {
            Ok(a) => a,
            Err(e) => {
                e.report(tcx, a.span, "expression");
                return Err(ErrorReported);
            }
        };
        let b = match self.eval(b) {
            Ok(b) => b,
            Err(e) => {
                e.report(tcx, b.span, "expression");
                return Err(ErrorReported);
            }
        };
        compare_const_vals(tcx, span, &a.val, &b.val)
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        const_eval,
        ..*providers
    };
}

fn const_eval<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                        -> EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let (def_id, substs) = if let Some(resolved) = lookup_const_by_id(tcx, key) {
        resolved
    } else {
        return Err(ConstEvalErr {
            span: tcx.def_span(key.value.0),
            kind: TypeckError
        });
    };

    let tables = tcx.typeck_tables_of(def_id);
    let body = if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        tcx.mir_const_qualif(def_id);
        tcx.hir.body(tcx.hir.body_owned_by(id))
    } else {
        tcx.extern_const_body(def_id).body
    };

    // do not continue into miri if typeck errors occurred
    // it will fail horribly
    if tables.tainted_by_errors {
        signal!(&body.value, TypeckError);
    }

    trace!("running old const eval");
    let old_result = ConstContext::new(tcx, key.param_env.and(substs), tables).eval(&body.value);
    trace!("old const eval produced {:?}", old_result);
    let instance = ty::Instance::new(def_id, substs);
    trace!("const eval instance: {:?}, {:?}", instance, key.param_env);
    let miri_result = ::rustc::mir::interpret::eval_body(tcx, instance, key.param_env);
    match (miri_result, old_result) {
        (Err(err), Ok(ok)) => {
            trace!("miri failed, ctfe returned {:?}", ok);
            tcx.sess.span_warn(
                tcx.def_span(key.value.0),
                "miri failed to eval, while ctfe succeeded",
            );
            let () = unwrap_miri(tcx, key.param_env, Err(err));
            Ok(ok)
        },
        (Ok(_), Err(err)) => {
            Err(err)
        },
        (Err(_), Err(err)) => Err(err),
        (Ok((miri_val, miri_ty)), Ok(ctfe)) => {
            check_ctfe_against_miri(tcx, key.param_env, miri_val, miri_ty, ctfe.val);
            Ok(ctfe)
        }
    }
}

fn check_ctfe_against_miri<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    miri_val: PtrAndAlign,
    miri_ty: Ty<'tcx>,
    ctfe: ConstVal<'tcx>,
) {
    let limits = ResourceLimits::default();
    use rustc::mir::interpret::{CompileTimeFunctionEvaluator, EvalContext, ResourceLimits};
    let mut ecx = EvalContext::<CompileTimeFunctionEvaluator>::new(tcx, limits, param_env, ());
    let value = ecx.read_maybe_aligned(miri_val.aligned, |ectx| {
        ectx.try_read_value(miri_val.ptr, miri_ty)
    });
    use rustc::ty::TypeVariants::*;
    match miri_ty.sty {
        TyInt(int_ty) => {
            let prim = get_prim(tcx, param_env, value);
            let c = ConstInt::new_signed_truncating(prim as i128,
                                                    int_ty,
                                                    tcx.sess.target.isize_ty);
            let c = ConstVal::Integral(c);
            assert_eq!(c, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", c, ctfe);
        },
        TyUint(uint_ty) => {
            let prim = get_prim(tcx, param_env, value);
            let c = ConstInt::new_unsigned_truncating(prim,
                                                     uint_ty,
                                                     tcx.sess.target.usize_ty);
            let c = ConstVal::Integral(c);
            assert_eq!(c, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", c, ctfe);
        },
        TyFloat(ty) => {
            let prim = get_prim(tcx, param_env, value);
            let f = ConstVal::Float(ConstFloat { bits: prim, ty });
            assert_eq!(f, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", f, ctfe);
        },
        TyBool => {
            let bits = get_prim(tcx, param_env, value);
            if bits > 1 {
                bug!("miri evaluated to {}, but expected a bool {:?}", bits, ctfe);
            }
            let b = ConstVal::Bool(bits == 1);
            assert_eq!(b, ctfe, "miri evaluated to {:?}, but ctfe yielded {:?}", b, ctfe);
        },
        TyChar => {
            let bits = get_prim(tcx, param_env, value);
            if let Some(cm) = ::std::char::from_u32(bits as u32) {
                assert_eq!(
                    ConstVal::Char(cm), ctfe,
                    "miri evaluated to {:?}, but expected {:?}", cm, ctfe,
                );
            } else {
                bug!("miri evaluated to {}, but expected a char {:?}", bits, ctfe);
            }
        },
        TyStr => {
            if let Ok(Some(Value::ByValPair(PrimVal::Ptr(ptr), PrimVal::Bytes(len)))) = value {
                let bytes = ecx
                    .memory
                    .read_bytes(ptr.into(), len as u64)
                    .expect("bad miri memory for str");
                if let Ok(s) = ::std::str::from_utf8(bytes) {
                    if let ConstVal::Str(s2) = ctfe {
                        assert_eq!(s, s2, "miri produced {:?}, but expected {:?}", s, s2);
                    } else {
                        bug!("miri produced {:?}, but expected {:?}", s, ctfe);
                    }
                } else {
                    bug!(
                        "miri failed to produce valid utf8 {:?}, while ctfe produced {:?}",
                        bytes,
                        ctfe,
                    );
                }
            } else {
                bug!("miri evaluated to {:?}, but expected a str {:?}", value, ctfe);
            }
        },
        TyArray(elem_ty, n) => {
            let n = n.val.to_const_int().unwrap().to_u64().unwrap();
            let size = ecx.type_size(elem_ty).unwrap().unwrap();
            let vec: Box<Iterator<Item = (ConstVal, Ty<'tcx>)>> = match ctfe {
                ConstVal::ByteStr(arr) => Box::new(arr.data.iter().map(|&b| {
                    (ConstVal::Integral(ConstInt::U8(b)), tcx.types.u8)
                })),
                ConstVal::Aggregate(Array(v)) => {
                    Box::new(v.iter().map(|c| (c.val, c.ty)))
                },
                ConstVal::Aggregate(Repeat(v, n)) => {
                    Box::new(::std::iter::repeat((v.val, v.ty)).take(n as usize))
                },
                _ => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            for (i, elem) in vec.enumerate() {
                assert!((i as u64) < n);
                let ptr = miri_val.offset(size * i as u64, &ecx).unwrap();
                check_ctfe_against_miri(tcx, param_env, ptr, elem_ty, elem.0);
            }
        },
        TyTuple(..) => {
            let vec = match ctfe {
                ConstVal::Aggregate(Tuple(v)) => v,
                _ => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            for (i, elem) in vec.into_iter().enumerate() {
                let offset = ecx.get_field_offset(miri_ty, i).unwrap();
                let ptr = miri_val.offset(offset.bytes(), &ecx).unwrap();
                check_ctfe_against_miri(tcx, param_env, ptr, elem.ty, elem.val);
            }
        },
        TyAdt(def, substs) => {
            let (struct_variant, extra) = if def.is_enum() {
                let ptr = miri_val.ptr.to_ptr().unwrap();
                let discr = ecx.read_discriminant_value(ptr, miri_ty).unwrap();
                let variant = def.discriminants(tcx).position(|variant_discr| {
                    variant_discr.to_u128_unchecked() == discr
                }).expect("miri produced invalid enum discriminant");
                (&def.variants[variant], LvalueExtra::DowncastVariant(variant))
            } else {
                (def.struct_variant(), LvalueExtra::None)
            };
            let vec = match ctfe {
                ConstVal::Aggregate(Struct(v)) => v,
                ConstVal::Variant(did) => {
                    assert_eq!(struct_variant.fields.len(), 0);
                    assert_eq!(did, struct_variant.did);
                    return;
                },
                ctfe => bug!("miri produced {:?}, but ctfe yielded {:?}", miri_ty, ctfe),
            };
            for &(name, elem) in vec.into_iter() {
                let field = struct_variant.fields.iter().position(|f| f.name == name).unwrap();
                let lvalue = ecx.lvalue_field(
                    Lvalue::Ptr { ptr: miri_val, extra },
                    Field::new(field),
                    miri_ty,
                    struct_variant.fields[field].ty(tcx, substs),
                ).unwrap();
                let ptr = lvalue.to_ptr_extra_aligned().0;
                check_ctfe_against_miri(tcx, param_env, ptr, elem.ty, elem.val);
            }
        },
        TySlice(_) => bug!("miri produced a slice?"),
        // not supported by ctfe
        TyRawPtr(_) |
        TyRef(..) => {}
        TyDynamic(..) => bug!("miri produced a trait object"),
        TyClosure(..) => bug!("miri produced a closure"),
        TyGenerator(..) => bug!("miri produced a generator"),
        TyNever => bug!("miri produced a value of the never type"),
        TyProjection(_) => bug!("miri produced a projection"),
        TyAnon(..) => bug!("miri produced an impl Trait type"),
        TyParam(_) => bug!("miri produced an unmonomorphized type"),
        TyInfer(_) => bug!("miri produced an uninferred type"),
        TyError => bug!("miri produced a type error"),
        // should be fine
        TyFnDef(..) => {}
        TyFnPtr(_) => {
            let ptr = match value {
                Ok(Some(Value::ByVal(PrimVal::Ptr(ptr)))) => ptr,
                value => bug!("expected fn ptr, got {:?}", value),
            };
            let inst = ecx.memory.get_fn(ptr).unwrap();
            match ctfe {
                ConstVal::Function(did, substs) => {
                    let ctfe = ty::Instance::resolve(ecx.tcx, param_env, did, substs).unwrap();
                    assert_eq!(inst, ctfe, "expected fn ptr {:?}, but got {:?}", ctfe, inst);
                },
                _ => bug!("ctfe produced {:?}, but miri produced function {:?}", ctfe, inst),
            }
        },
    }
}

fn get_prim<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    res: Result<Option<Value>, EvalError<'tcx>>,
) -> u128 {
    match res {
        Ok(Some(Value::ByVal(prim))) => unwrap_miri(tcx, param_env, prim.to_bytes()),
        Err(err) => unwrap_miri(tcx, param_env, Err(err)),
        val => bug!("got {:?}", val),
    }
}

fn unwrap_miri<'a, 'tcx, T>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    res: Result<T, EvalError<'tcx>>,
) -> T {
    match res {
        Ok(val) => val,
        Err(mut err) => {
            let limits = ResourceLimits::default();
            use rustc::mir::interpret::{CompileTimeFunctionEvaluator, EvalContext, ResourceLimits};
            let ecx = EvalContext::<CompileTimeFunctionEvaluator>::new(tcx, limits, param_env, ());
            ecx.report(&mut err);
            tcx.sess.abort_if_errors();
            bug!("{:#?}", err);
        }
    }
}
