// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! This module defines the `DepNode` type which the compiler uses to represent
//! nodes in the dependency graph. A `DepNode` consists of a `DepKind` (which
//! specifies the kind of thing it represents, like a piece of HIR, MIR, etc)
//! and a `Fingerprint`, a 128 bit hash value the exact meaning of which
//! depends on the node's `DepKind`. Together, the kind and the fingerprint
//! fully identify a dependency node, even across multiple compilation sessions.
//! In other words, the value of the fingerprint does not depend on anything
//! that is specific to a given compilation session, like an unpredictable
//! interning key (e.g. NodeId, DefId, Symbol) or the numeric value of a
//! pointer. The concept behind this could be compared to how git commit hashes
//! uniquely identify a given commit and has a few advantages:
//!
//! * A `DepNode` can simply be serialized to disk and loaded in another session
//!   without the need to do any "rebasing (like we have to do for Spans and
//!   NodeIds) or "retracing" like we had to do for `DefId` in earlier
//!   implementations of the dependency graph.
//! * A `Fingerprint` is just a bunch of bits, which allows `DepNode` to
//!   implement `Copy`, `Sync`, `Send`, `Freeze`, etc.
//! * Since we just have a bit pattern, `DepNode` can be mapped from disk into
//!   memory without any post-processing (e.g. "abomination-style" pointer
//!   reconstruction).
//! * Because a `DepNode` is self-contained, we can instantiate `DepNodes` that
//!   refer to things that do not exist anymore. In previous implementations
//!   `DepNode` contained a `DefId`. A `DepNode` referring to something that
//!   had been removed between the previous and the current compilation session
//!   could not be instantiated because the current compilation session
//!   contained no `DefId` for thing that had been removed.
//!
//! `DepNode` definition happens in the `define_dep_nodes!()` macro. This macro
//! defines the `DepKind` enum and a corresponding `DepConstructor` enum. The
//! `DepConstructor` enum links a `DepKind` to the parameters that are needed at
//! runtime in order to construct a valid `DepNode` fingerprint.
//!
//! Because the macro sees what parameters a given `DepKind` requires, it can
//! "infer" some properties for each kind of `DepNode`:
//!
//! * Whether a `DepNode` of a given kind has any parameters at all. Some
//!   `DepNode`s, like `Krate`, represent global concepts with only one value.
//! * Whether it is possible, in principle, to reconstruct a query key from a
//!   given `DepNode`. Many `DepKind`s only require a single `DefId` parameter,
//!   in which case it is possible to map the node's fingerprint back to the
//!   `DefId` it was computed from. In other cases, too much information gets
//!   lost during fingerprint computation.
//!
//! The `DepConstructor` enum, together with `DepNode::new()` ensures that only
//! valid `DepNode` instances can be constructed. For example, the API does not
//! allow for constructing parameterless `DepNode`s with anything other
//! than a zeroed out fingerprint. More generally speaking, it relieves the
//! user of the `DepNode` API of having to know how to compute the expected
//! fingerprint for a given set of node parameters.

use hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX};
use hir::map::DefPathHash;
use hir::{HirId, ItemLocalId};

use ich::Fingerprint;
use ty::{TyCtxt, Instance, InstanceDef, ParamEnvAnd, Ty};
use ty::subst::Substs;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use ich::StableHashingContext;
use std::fmt;
use std::hash::Hash;
use syntax_pos::symbol::InternedString;

// erase!() just makes tokens go away. It's used to specify which macro argument
// is repeated (i.e. which sub-expression of the macro we are in) but don't need
// to actually use any of the arguments.
macro_rules! erase {
    ($x:tt) => ({})
}

macro_rules! is_anon_attr {
    (anon) => (true);
    ($attr:ident) => (false);
}

macro_rules! is_input_attr {
    (input) => (true);
    ($attr:ident) => (false);
}

macro_rules! contains_anon_attr {
    ($($attr:ident),*) => ({$(is_anon_attr!($attr) | )* false});
}

macro_rules! contains_input_attr {
    ($($attr:ident),*) => ({$(is_input_attr!($attr) | )* false});
}

macro_rules! define_dep_nodes {
    (<$tcx:tt>
    $(
        [$($attr:ident),* ]
        $variant:ident $(( $($tuple_arg:tt),* ))*
                       $({ $($struct_arg_name:ident : $struct_arg_ty:ty),* })*
      ,)*
    ) => (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub enum DepKind {
            $($variant),*
        }

        impl DepKind {
            #[allow(unreachable_code)]
            #[inline]
            pub fn can_reconstruct_query_key<$tcx>(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            if contains_anon_attr!($($attr),*) {
                                return false;
                            }

                            // tuple args
                            $({
                                return <( $($tuple_arg,)* ) as DepNodeParams>
                                    ::CAN_RECONSTRUCT_QUERY_KEY;
                            })*

                            // struct args
                            $({

                                return <( $($struct_arg_ty,)* ) as DepNodeParams>
                                    ::CAN_RECONSTRUCT_QUERY_KEY;
                            })*

                            true
                        }
                    )*
                }
            }

            #[inline]
            pub fn is_anon(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_anon_attr!($($attr),*) }
                    )*
                }
            }

            #[inline]
            pub fn is_input(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => { contains_input_attr!($($attr),*) }
                    )*
                }
            }

            #[allow(unreachable_code)]
            #[inline]
            pub fn has_params(&self) -> bool {
                match *self {
                    $(
                        DepKind :: $variant => {
                            // tuple args
                            $({
                                $(erase!($tuple_arg);)*
                                return true;
                            })*

                            // struct args
                            $({
                                $(erase!($struct_arg_name);)*
                                return true;
                            })*

                            false
                        }
                    )*
                }
            }
        }

        pub enum DepConstructor<$tcx> {
            $(
                $variant $(( $($tuple_arg),* ))*
                         $({ $($struct_arg_name : $struct_arg_ty),* })*
            ),*
        }

        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
                 RustcEncodable, RustcDecodable)]
        pub struct DepNode {
            pub kind: DepKind,
            pub hash: Fingerprint,
        }

        impl DepNode {
            #[allow(unreachable_code, non_snake_case)]
            pub fn new<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                       dep: DepConstructor<'gcx>)
                                       -> DepNode
                where 'gcx: 'a + 'tcx,
                      'tcx: 'a
            {
                match dep {
                    $(
                        DepConstructor :: $variant $(( $($tuple_arg),* ))*
                                                   $({ $($struct_arg_name),* })*
                            =>
                        {
                            // tuple args
                            $({
                                let tupled_args = ( $($tuple_arg,)* );
                                let hash = DepNodeParams::to_fingerprint(&tupled_args,
                                                                         tcx);
                                let dep_node = DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };

                                if cfg!(debug_assertions) &&
                                   !dep_node.kind.can_reconstruct_query_key() &&
                                   (tcx.sess.opts.debugging_opts.incremental_info ||
                                    tcx.sess.opts.debugging_opts.query_dep_graph)
                                {
                                    tcx.dep_graph.register_dep_node_debug_str(dep_node, || {
                                        tupled_args.to_debug_str(tcx)
                                    });
                                }

                                return dep_node;
                            })*

                            // struct args
                            $({
                                let tupled_args = ( $($struct_arg_name,)* );
                                let hash = DepNodeParams::to_fingerprint(&tupled_args,
                                                                         tcx);
                                let dep_node = DepNode {
                                    kind: DepKind::$variant,
                                    hash
                                };

                                if cfg!(debug_assertions) &&
                                   !dep_node.kind.can_reconstruct_query_key() &&
                                   (tcx.sess.opts.debugging_opts.incremental_info ||
                                    tcx.sess.opts.debugging_opts.query_dep_graph)
                                {
                                    tcx.dep_graph.register_dep_node_debug_str(dep_node, || {
                                        tupled_args.to_debug_str(tcx)
                                    });
                                }

                                return dep_node;
                            })*

                            DepNode {
                                kind: DepKind::$variant,
                                hash: Fingerprint::zero(),
                            }
                        }
                    )*
                }
            }

            /// Construct a DepNode from the given DepKind and DefPathHash. This
            /// method will assert that the given DepKind actually requires a
            /// single DefId/DefPathHash parameter.
            #[inline]
            pub fn from_def_path_hash(kind: DepKind,
                                      def_path_hash: DefPathHash)
                                      -> DepNode {
                assert!(kind.can_reconstruct_query_key() && kind.has_params());
                DepNode {
                    kind,
                    hash: def_path_hash.0,
                }
            }

            /// Create a new, parameterless DepNode. This method will assert
            /// that the DepNode corresponding to the given DepKind actually
            /// does not require any parameters.
            #[inline]
            pub fn new_no_params(kind: DepKind) -> DepNode {
                assert!(!kind.has_params());
                DepNode {
                    kind,
                    hash: Fingerprint::zero(),
                }
            }

            /// Extract the DefId corresponding to this DepNode. This will work
            /// if two conditions are met:
            ///
            /// 1. The Fingerprint of the DepNode actually is a DefPathHash, and
            /// 2. the item that the DefPath refers to exists in the current tcx.
            ///
            /// Condition (1) is determined by the DepKind variant of the
            /// DepNode. Condition (2) might not be fulfilled if a DepNode
            /// refers to something from the previous compilation session that
            /// has been removed.
            #[inline]
            pub fn extract_def_id(&self, tcx: TyCtxt) -> Option<DefId> {
                if self.kind.can_reconstruct_query_key() {
                    let def_path_hash = DefPathHash(self.hash);
                    if let Some(ref def_path_map) = tcx.def_path_hash_to_def_id.as_ref() {
                        def_path_map.get(&def_path_hash).cloned()
                    } else {
                       None
                    }
                } else {
                    None
                }
            }

            /// Used in testing
            pub fn from_label_string(label: &str,
                                     def_path_hash: DefPathHash)
                                     -> Result<DepNode, ()> {
                let kind = match label {
                    $(
                        stringify!($variant) => DepKind::$variant,
                    )*
                    _ => return Err(()),
                };

                if !kind.can_reconstruct_query_key() {
                    return Err(());
                }

                if kind.has_params() {
                    Ok(def_path_hash.to_dep_node(kind))
                } else {
                    Ok(DepNode::new_no_params(kind))
                }
            }
        }
    );
}

impl fmt::Debug for DepNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.kind)?;

        if !self.kind.has_params() && !self.kind.is_anon() {
            return Ok(());
        }

        write!(f, "(")?;

        ::ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                if let Some(def_id) = self.extract_def_id(tcx) {
                    write!(f, "{}", tcx.def_path(def_id).to_string(tcx))?;
                } else if let Some(ref s) = tcx.dep_graph.dep_node_debug_str(*self) {
                    write!(f, "{}", s)?;
                } else {
                    write!(f, "{}", self.hash)?;
                }
            } else {
                write!(f, "{}", self.hash)?;
            }
            Ok(())
        })?;

        write!(f, ")")
    }
}


impl DefPathHash {
    #[inline]
    pub fn to_dep_node(self, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, self)
    }
}

impl DefId {
    #[inline]
    pub fn to_dep_node(self, tcx: TyCtxt, kind: DepKind) -> DepNode {
        DepNode::from_def_path_hash(kind, tcx.def_path_hash(self))
    }
}

impl DepKind {
    #[inline]
    pub fn fingerprint_needed_for_crate_hash(self) -> bool {
        match self {
            DepKind::HirBody |
            DepKind::Krate => true,
            _ => false,
        }
    }
}

define_dep_nodes!( <'tcx>
    // Represents the `Krate` as a whole (the `hir::Krate` value) (as
    // distinct from the krate module). This is basically a hash of
    // the entire krate, so if you read from `Krate` (e.g., by calling
    // `tcx.hir.krate()`), we will have to assume that any change
    // means that you need to be recompiled. This is because the
    // `Krate` value gives you access to all other items. To avoid
    // this fate, do not call `tcx.hir.krate()`; instead, prefer
    // wrappers like `tcx.visit_all_items_in_krate()`.  If there is no
    // suitable wrapper, you can use `tcx.dep_graph.ignore()` to gain
    // access to the krate, but you must remember to add suitable
    // edges yourself for the individual items that you read.
    [input] Krate,

    // Represents the body of a function or method. The def-id is that of the
    // function/method.
    [input] HirBody(DefId),

    // Represents the HIR node with the given node-id
    [input] Hir(DefId),

    // Represents metadata from an extern crate.
    [input] CrateMetadata(CrateNum),

    // Represents some artifact that we save to disk. Note that these
    // do not have a def-id as part of their identifier.
    [] WorkProduct(WorkProductId),

    // Represents different phases in the compiler.
    [] RegionScopeTree(DefId),
    [] Coherence,
    [] CoherenceInherentImplOverlapCheck,
    [] CoherenceCheckTrait(DefId),
    [] PrivacyAccessLevels(CrateNum),

    // Represents the MIR for a fn; also used as the task node for
    // things read/modify that MIR.
    [] MirConstQualif(DefId),
    [] MirConst(DefId),
    [] MirValidated(DefId),
    [] MirOptimized(DefId),
    [] MirShim { instance_def: InstanceDef<'tcx> },

    [] BorrowCheckKrate,
    [] BorrowCheck(DefId),
    [] MirBorrowCheck(DefId),
    [] UnsafetyViolations(DefId),

    [] Reachability,
    [] MirKeys,
    [] CrateVariances,

    // Nodes representing bits of computed IR in the tcx. Each shared
    // table in the tcx (or elsewhere) maps to one of these
    // nodes.
    [] AssociatedItems(DefId),
    [] TypeOfItem(DefId),
    [] GenericsOfItem(DefId),
    [] PredicatesOfItem(DefId),
    [] SuperPredicatesOfItem(DefId),
    [] TraitDefOfItem(DefId),
    [] AdtDefOfItem(DefId),
    [] IsDefaultImpl(DefId),
    [] ImplTraitRef(DefId),
    [] ImplPolarity(DefId),
    [] ClosureKind(DefId),
    [] FnSignature(DefId),
    [] GenSignature(DefId),
    [] CoerceUnsizedInfo(DefId),

    [] ItemVarianceConstraints(DefId),
    [] ItemVariances(DefId),
    [] IsConstFn(DefId),
    [] IsForeignItem(DefId),
    [] TypeParamPredicates { item_id: DefId, param_id: DefId },
    [] SizedConstraint(DefId),
    [] DtorckConstraint(DefId),
    [] AdtDestructor(DefId),
    [] AssociatedItemDefIds(DefId),
    [] InherentImpls(DefId),
    [] TypeckBodiesKrate,
    [] TypeckTables(DefId),
    [] HasTypeckTables(DefId),
    [] ConstEval { param_env: ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)> },
    [] SymbolName(DefId),
    [] InstanceSymbolName { instance: Instance<'tcx> },
    [] SpecializationGraph(DefId),
    [] ObjectSafety(DefId),

    [] IsCopy { param_env: ParamEnvAnd<'tcx, Ty<'tcx>> },
    [] IsSized { param_env: ParamEnvAnd<'tcx, Ty<'tcx>> },
    [] IsFreeze { param_env: ParamEnvAnd<'tcx, Ty<'tcx>> },
    [] NeedsDrop { param_env: ParamEnvAnd<'tcx, Ty<'tcx>> },
    [] Layout { param_env: ParamEnvAnd<'tcx, Ty<'tcx>> },

    // The set of impls for a given trait.
    [] TraitImpls(DefId),

    [] AllLocalTraitImpls,

    // Trait selection cache is a little funny. Given a trait
    // reference like `Foo: SomeTrait<Bar>`, there could be
    // arbitrarily many def-ids to map on in there (e.g., `Foo`,
    // `SomeTrait`, `Bar`). We could have a vector of them, but it
    // requires heap-allocation, and trait sel in general can be a
    // surprisingly hot path. So instead we pick two def-ids: the
    // trait def-id, and the first def-id in the input types. If there
    // is no def-id in the input types, then we use the trait def-id
    // again. So for example:
    //
    // - `i32: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `u32: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `Clone: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Clone }`
    // - `Vec<i32>: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: Vec }`
    // - `String: Clone` -> `TraitSelect { trait_def_id: Clone, self_def_id: String }`
    // - `Foo: Trait<Bar>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `Foo: Trait<i32>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `(Foo, Bar): Trait` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    // - `i32: Trait<Foo>` -> `TraitSelect { trait_def_id: Trait, self_def_id: Foo }`
    //
    // You can see that we map many trait refs to the same
    // trait-select node.  This is not a problem, it just means
    // imprecision in our dep-graph tracking.  The important thing is
    // that for any given trait-ref, we always map to the **same**
    // trait-select node.
    [anon] TraitSelect,

    [] ParamEnv(DefId),
    [] DescribeDef(DefId),
    [] DefSpan(DefId),
    [] LookupStability(DefId),
    [] LookupDeprecationEntry(DefId),
    [] ItemBodyNestedBodies(DefId),
    [] ConstIsRvaluePromotableToStatic(DefId),
    [] ImplParent(DefId),
    [] TraitOfItem(DefId),
    [] IsExportedSymbol(DefId),
    [] IsMirAvailable(DefId),
    [] ItemAttrs(DefId),
    [] FnArgNames(DefId),
    [] DylibDepFormats(CrateNum),
    [] IsPanicRuntime(CrateNum),
    [] IsCompilerBuiltins(CrateNum),
    [] HasGlobalAllocator(CrateNum),
    [] ExternCrate(DefId),
    [] LintLevels,
    [] Specializes { impl1: DefId, impl2: DefId },
    [input] InScopeTraits(DefIndex),
    [] ModuleExports(DefId),
    [] IsSanitizerRuntime(CrateNum),
    [] IsProfilerRuntime(CrateNum),
    [] GetPanicStrategy(CrateNum),
    [] IsNoBuiltins(CrateNum),
    [] ImplDefaultness(DefId),
    [] ExportedSymbolIds(CrateNum),
    [] NativeLibraries(CrateNum),
    [] PluginRegistrarFn(CrateNum),
    [] DeriveRegistrarFn(CrateNum),
    [] CrateDisambiguator(CrateNum),
    [] CrateHash(CrateNum),
    [] OriginalCrateName(CrateNum),

    [] ImplementationsOfTrait { krate: CrateNum, trait_id: DefId },
    [] AllTraitImplementations(CrateNum),

    [] IsDllimportForeignItem(DefId),
    [] IsStaticallyIncludedForeignItem(DefId),
    [] NativeLibraryKind(DefId),
    [] LinkArgs,

    [] NamedRegion(DefIndex),
    [] IsLateBound(DefIndex),
    [] ObjectLifetimeDefaults(DefIndex),

    [] Visibility(DefId),
    [] DepKind(CrateNum),
    [] CrateName(CrateNum),
    [] ItemChildren(DefId),
    [] ExternModStmtCnum(DefId),
    [] GetLangItems,
    [] DefinedLangItems(CrateNum),
    [] MissingLangItems(CrateNum),
    [] ExternConstBody(DefId),
    [] VisibleParentMap,
    [] MissingExternCrateItem(CrateNum),
    [] UsedCrateSource(CrateNum),
    [] PostorderCnums,
    [] HasCloneClosures(CrateNum),
    [] HasCopyClosures(CrateNum),

    [] Freevars(DefId),
    [] MaybeUnusedTraitImport(DefId),
    [] MaybeUnusedExternCrates,
    [] StabilityIndex,
    [] AllCrateNums,
    [] ExportedSymbols(CrateNum),
    [] CollectAndPartitionTranslationItems,
    [] ExportName(DefId),
    [] ContainsExternIndicator(DefId),
    [] IsTranslatedFunction(DefId),
    [] CodegenUnit(InternedString),
    [] CompileCodegenUnit(InternedString),
    [] OutputFilenames,

    // We use this for most things when incr. comp. is turned off.
    [] Null,
);

trait DepNodeParams<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> : fmt::Debug {
    const CAN_RECONSTRUCT_QUERY_KEY: bool;

    /// This method turns the parameters of a DepNodeConstructor into an opaque
    /// Fingerprint to be used in DepNode.
    /// Not all DepNodeParams support being turned into a Fingerprint (they
    /// don't need to if the corresponding DepNode is anonymous).
    fn to_fingerprint(&self, _: TyCtxt<'a, 'gcx, 'tcx>) -> Fingerprint {
        panic!("Not implemented. Accidentally called on anonymous node?")
    }

    fn to_debug_str(&self, _: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        format!("{:?}", self)
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a, T> DepNodeParams<'a, 'gcx, 'tcx> for T
    where T: HashStable<StableHashingContext<'gcx>> + fmt::Debug
{
    default const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    default fn to_fingerprint(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Fingerprint {
        let mut hcx = tcx.create_stable_hashing_context();
        let mut hasher = StableHasher::new();

        self.hash_stable(&mut hcx, &mut hasher);

        hasher.finish()
    }

    default fn to_debug_str(&self, _: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        format!("{:?}", *self)
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (DefId,) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        tcx.def_path_hash(self.0).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        tcx.item_path_str(self.0)
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (DefIndex,) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        tcx.hir.definitions().def_path_hash(self.0).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        tcx.item_path_str(DefId::local(self.0))
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (CrateNum,) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = true;

    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        let def_id = DefId {
            krate: self.0,
            index: CRATE_DEF_INDEX,
        };
        tcx.def_path_hash(def_id).0
    }

    fn to_debug_str(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        tcx.crate_name(self.0).as_str().to_string()
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (DefId, DefId) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }

    fn to_debug_str(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> String {
        let (def_id_0, def_id_1) = *self;

        format!("({}, {})",
                tcx.def_path(def_id_0).to_string(tcx),
                tcx.def_path(def_id_1).to_string(tcx))
    }
}

impl<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> DepNodeParams<'a, 'gcx, 'tcx> for (HirId,) {
    const CAN_RECONSTRUCT_QUERY_KEY: bool = false;

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    fn to_fingerprint(&self, tcx: TyCtxt) -> Fingerprint {
        let (HirId {
            owner,
            local_id: ItemLocalId(local_id),
        },) = *self;

        let def_path_hash = tcx.def_path_hash(DefId::local(owner));
        let local_id = Fingerprint::from_smaller_hash(local_id as u64);

        def_path_hash.0.combine(local_id)
    }
}

/// A "work product" corresponds to a `.o` (or other) file that we
/// save in between runs. These ids do not have a DefId but rather
/// some independent path or string that persists between runs without
/// the need to be mapped or unmapped. (This ensures we can serialize
/// them even in the absence of a tcx.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
         RustcEncodable, RustcDecodable)]
pub struct WorkProductId {
    hash: Fingerprint
}

impl WorkProductId {
    pub fn from_cgu_name(cgu_name: &str) -> WorkProductId {
        let mut hasher = StableHasher::new();
        cgu_name.len().hash(&mut hasher);
        cgu_name.hash(&mut hasher);
        WorkProductId {
            hash: hasher.finish()
        }
    }

    pub fn from_fingerprint(fingerprint: Fingerprint) -> WorkProductId {
        WorkProductId {
            hash: fingerprint
        }
    }

    pub fn to_dep_node(self) -> DepNode {
        DepNode {
            kind: DepKind::WorkProduct,
            hash: self.hash,
        }
    }
}

impl_stable_hash_for!(struct ::dep_graph::WorkProductId {
    hash
});

