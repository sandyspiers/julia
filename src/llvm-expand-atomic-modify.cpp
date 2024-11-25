// This file is a part of Julia. License is MIT: https://julialang.org/license

// TODO: move this feature into AtomicExpandImpl

#include "llvm-version.h"
#include "passes.h"

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include <llvm/Analysis/InstSimplifyFolder.h>
#include <llvm/CodeGen/AtomicExpandUtils.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LowerAtomic.h>

#include "julia.h"
#include "julia_assert.h"

#define DEBUG_TYPE "expand-atomic-modify"
#undef DEBUG

using namespace llvm;

// This pass takes fake call instructions that look like this which were emitted by the front end:
//   (oldval, newval) = call atomicmodify.iN(ptr %op, ptr align(N) %ptr, i8 immarg %SSID, i8 immarg %Ordering, ...) !rmwattributes
//   where op is a function with a prototype of `iN (iN arg, ...)`
// Then rewrite that to
//   oldval = atomicrmw op ptr, val ordering syncscope
//   newval = op oldval, val
// Or to an equivalent RMWCmpXchgLoop if `op` isn't valid for atomicrmw


// from AtomicExpandImpl, with modification of failure order and added Attributes
using CreateWeakCmpXchgInstFun =
   function_ref<void(IRBuilderBase &, Value *, Value *, Value *, Align,
                     AtomicOrdering, SyncScope::ID, Instruction &Attributes,
                     Value *&, Value *&)>;

static void createWeakCmpXchgInstFun(IRBuilderBase &Builder, Value *Addr,
                                 Value *Loaded, Value *NewVal, Align AddrAlign,
                                 AtomicOrdering MemOpOrder, SyncScope::ID SSID, Instruction &Attributes,
                                 Value *&Success, Value *&NewLoaded) {
  Type *OrigTy = NewVal->getType();

  // This code can go away when cmpxchg supports FP types.
  assert(!OrigTy->isPointerTy());
  bool NeedBitcast = OrigTy->isFloatingPointTy();
  if (NeedBitcast) {
    IntegerType *IntTy = Builder.getIntNTy(OrigTy->getPrimitiveSizeInBits());
    NewVal = Builder.CreateBitCast(NewVal, IntTy);
    Loaded = Builder.CreateBitCast(Loaded, IntTy);
  }

  AtomicCmpXchgInst *Pair = Builder.CreateAtomicCmpXchg(
      Addr, Loaded, NewVal, AddrAlign, MemOpOrder,
      AtomicOrdering::Monotonic, // why does LLVM use AtomicCmpXchgInst::getStrongestFailureOrdering(MemOpOrder) here
      SSID);
  Pair->copyMetadata(Attributes);
  Success = Builder.CreateExtractValue(Pair, 1, "success");
  NewLoaded = Builder.CreateExtractValue(Pair, 0, "newloaded");

  if (NeedBitcast)
    NewLoaded = Builder.CreateBitCast(NewLoaded, OrigTy);
}

// from AtomicExpandImpl, with modification of values returned
std::pair<Value *, Value *> insertRMWCmpXchgLoop(
    IRBuilderBase &Builder, Type *ResultTy, Value *Addr, Align AddrAlign,
    AtomicOrdering MemOpOrder, SyncScope::ID SSID, Instruction &Attributes,
    function_ref<Value *(IRBuilderBase &, Value *)> PerformOp,
    CreateWeakCmpXchgInstFun CreateWeakCmpXchg) {
  LLVMContext &Ctx = Builder.getContext();
  BasicBlock *BB = Builder.GetInsertBlock();
  Function *F = BB->getParent();

  // Given: atomicrmw some_op iN* %addr, iN %incr ordering
  //
  // The standard expansion we produce is:
  //     [...]
  //     %init_loaded = load atomic iN* %addr
  //     br label %loop
  // loop:
  //     %loaded = phi iN [ %init_loaded, %entry ], [ %new_loaded, %loop ]
  //     %new = some_op iN %loaded, %incr
  //     %pair = cmpxchg iN* %addr, iN %loaded, iN %new
  //     %new_loaded = extractvalue { iN, i1 } %pair, 0
  //     %success = extractvalue { iN, i1 } %pair, 1
  //     br i1 %success, label %atomicrmw.end, label %loop
  // atomicrmw.end:
  //     [...]
  BasicBlock *ExitBB =
      BB->splitBasicBlock(Builder.GetInsertPoint(), "atomicrmw.end");
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we want a load. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  LoadInst *InitLoaded = Builder.CreateAlignedLoad(ResultTy, Addr, AddrAlign);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  PHINode *Loaded = Builder.CreatePHI(ResultTy, 2, "loaded");
  Loaded->addIncoming(InitLoaded, BB);

  Value *NewVal = PerformOp(Builder, Loaded);

  Value *NewLoaded = nullptr;
  Value *Success = nullptr;

  CreateWeakCmpXchg(Builder, Addr, Loaded, NewVal, AddrAlign,
                MemOpOrder == AtomicOrdering::Unordered
                    ? AtomicOrdering::Monotonic
                    : MemOpOrder,
                SSID, Attributes, Success, NewLoaded);
  assert(Success && NewLoaded);

  Loaded->addIncoming(NewLoaded, LoopBB);

  Builder.CreateCondBr(Success, ExitBB, LoopBB);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());
  return {NewLoaded, NewVal};
}

// from AtomicExpandImpl
struct ReplacementIRBuilder : IRBuilder<InstSimplifyFolder> {
  // Preserves the DebugLoc from I, and preserves still valid metadata.
  explicit ReplacementIRBuilder(Instruction *I, const DataLayout &DL)
      : IRBuilder(I->getContext(), DL) {
    SetInsertPoint(I);
    this->CollectMetadataToCopy(I, {LLVMContext::MD_pcsections});
  }
};

// Must check that either Target cannot observe or mutate global state
// or that no trailing instructions does so either.
// Depending on the choice, it can also decide whether it is better to move Target after RMW
// or to move RMW before Target (or meet somewhere in the middle).
// Currently conservatively implemented as there being no instruction in the
// function which writes memory (which includes any atomics).
// Excluding the Target itself, unless some other instruction might read memory to observe it.
static bool canReorderWithRMW(Instruction &Target, bool verifyop)
{
  if (!verifyop)
    return true;
  Function &Op = *Target.getFunction();
  // quick check: if Op is nosync and Target doesn't access any memory, then reordering is trivially valid
  bool nosync = Op.hasNoSync();
  if (nosync && !Target.mayReadOrWriteMemory())
    return true;
  // otherwise, scan the whole function to see if any function accesses memory
  // in a way that would conflict with reordering the atomic read and write
  bool mayRead = false;
  for (auto &BB : Op) {
    for (auto &I : BB) {
      if (&I == &Target)
        continue;
      if (I.mayWriteToMemory())
        return false;
      if (!mayRead) {
        mayRead = I.mayReadFromMemory();
        if (!nosync && mayRead)
          return false;
      }
    }
  }
  // if any other instruction read memory, then the ordering of any writes by the target instruction might be observed
  return !(mayRead && Target.mayWriteToMemory());
}

static AtomicRMWInst::BinOp patternMatchAtomicRMWOp(Value *Old, unsigned &ValOp, bool &tryinline, Value *RetVal)
{
  ValOp = -1u; // special sentinal
  tryinline = false;
  bool verifyop = RetVal == nullptr;
  assert(verifyop ? isa<Argument>(Old) : isa<AtomicRMWInst>(Old));
  Function *Op = verifyop ? cast<Argument>(Old)->getParent() : nullptr;
  if (verifyop && (Op->isDeclaration() || Op->isInterposable() || Op->isIntrinsic()))
    return AtomicRMWInst::BAD_BINOP;
   // TODO: peek forward from Old through any trivial casts which don't affect the instruction (e.g. i64 to f64 and back)
  if (RetVal == nullptr) {
    if (Old->use_empty())
      return AtomicRMWInst::Xchg;
    if (!Old->hasOneUse())
      return AtomicRMWInst::BAD_BINOP;
    ReturnInst *Ret = nullptr;
    for (auto &BB : *Op) {
      if (isa<ReturnInst>(BB.getTerminator())) {
        if (Ret != nullptr)
          return AtomicRMWInst::BAD_BINOP;
        Ret = cast<ReturnInst>(BB.getTerminator());
      }
    }
    if (Ret == nullptr)
      return AtomicRMWInst::BAD_BINOP;
    // Now examine the instruction list
    RetVal = Ret->getReturnValue();
    if (!RetVal->hasOneUse())
      return AtomicRMWInst::BAD_BINOP;
  }
  if (RetVal == Old) {
    // special token indicating to convert to an atomic fence
    return AtomicRMWInst::Or;
  }
  if (Old->use_empty())
    return AtomicRMWInst::Xchg;
  if (auto BinOp = dyn_cast<BinaryOperator>(RetVal)) {
    if ((BinOp->getOperand(0) == Old || (BinOp->isCommutative() && BinOp->getOperand(1) == Old)) && canReorderWithRMW(*BinOp, verifyop)) {
      ValOp = BinOp->getOperand(0) == Old ? 1 : 0;
      switch (BinOp->getOpcode()) {
        case Instruction::Add:
          return AtomicRMWInst::Add;
        case Instruction::Sub:
          return AtomicRMWInst::Sub;
        case Instruction::And:
          return AtomicRMWInst::And;
        case Instruction::Or:
          return AtomicRMWInst::Or;
        case Instruction::Xor:
          return AtomicRMWInst::Xor;
        case Instruction::FAdd:
          return AtomicRMWInst::FAdd;
        case Instruction::FSub:
          return AtomicRMWInst::FSub;
        default:
          break;
      }
    }
    return AtomicRMWInst::BAD_BINOP;
  } else if (auto Intr = dyn_cast<IntrinsicInst>(RetVal)) {
    if (Intr->arg_size() == 2) {
      if ((Intr->getOperand(0) == Old || (Intr->isCommutative() && Intr->getOperand(1) == Old)) && canReorderWithRMW(*Intr, verifyop)) {
        ValOp = Intr->getOperand(0) == Old ? 1 : 0;
        switch (Intr->getIntrinsicID()) {
          case Intrinsic::minnum:
            return AtomicRMWInst::FMin;
          case Intrinsic::maxnum:
            return AtomicRMWInst::FMax;
        }
      }
    }
    return AtomicRMWInst::BAD_BINOP;
  }
  else if (auto Intr = dyn_cast<CallInst>(RetVal)) {
    // TODO: decide inlining cost of Op, or check alwaysinline/inlinehint, before this?
    for (unsigned OldArg = 0; OldArg < Intr->arg_size(); ++OldArg) {
      if (Intr->getArgOperandUse(OldArg) == Old) {
        if (canReorderWithRMW(*Intr, verifyop)) {
          tryinline = true;
          ValOp = OldArg;
        }
        return AtomicRMWInst::BAD_BINOP;
      }
    }
  }
  // TODO: does this need to deal with F->hasFnAttribute(Attribute::StrictFP)?
  // TODO: does Fneg and Neg have expansions?
  // TODO: be able to ignore some simple bitcasts (particularly f64 to i64)
  // TODO: handle longer sequences (Nand, Min, Max, UMax, UMin, UIncWrap, UDecWrap, and target-specific ones for CUDA)
  return AtomicRMWInst::BAD_BINOP;
}

void expandAtomicModifyToCmpXchg(CallInst &Modify,
                                 CreateWeakCmpXchgInstFun CreateWeakCmpXchg) {
  Value *Ptr = Modify.getOperand(0);
  Function *Op = cast<Function>(Modify.getOperand(1));
  AtomicOrdering Ordering = (AtomicOrdering)cast<ConstantInt>(Modify.getOperand(2))->getZExtValue();
  SyncScope::ID SSID = (SyncScope::ID)cast<ConstantInt>(Modify.getOperand(3))->getZExtValue();
  MaybeAlign Alignment = Modify.getParamAlign(0);
  unsigned user_arg_start = Modify.getFunctionType()->getNumParams();
  Type *Ty = Modify.getFunctionType()->getReturnType()->getStructElementType(0);

  ReplacementIRBuilder Builder(&Modify, Modify.getModule()->getDataLayout());
  Builder.setIsFPConstrained(Modify.hasFnAttr(Attribute::StrictFP));

  unsigned LoadedOp = 0;
  CallInst *ModifyOp;
  {
    SmallVector<Value*> Args(1 + Modify.arg_size() - user_arg_start);
    Args[LoadedOp] = UndefValue::get(Ty); // Undef used as placeholder for Loaded / RMW;
    for (size_t argi = 0; argi < Modify.arg_size() - user_arg_start; ++argi) {
      Args[argi + 1] = Modify.getArgOperand(argi + user_arg_start);
    }
    SmallVector<OperandBundleDef> Defs;
    Modify.getOperandBundlesAsDefs(Defs);
    ModifyOp = Builder.CreateCall(Op, Args, Defs);
    ModifyOp->setCallingConv(Op->getCallingConv());
  }

  Value *OldVal = nullptr;
  Value *NewVal = nullptr;
  unsigned ValOp;
  bool tryinline;
  auto BinOp = patternMatchAtomicRMWOp(Op->getArg(0), ValOp, tryinline, nullptr);
  if (tryinline || BinOp != AtomicRMWInst::BAD_BINOP) {
    Builder.SetInsertPoint(ModifyOp);
    AtomicRMWInst *RMW = Builder.CreateAtomicRMW(AtomicRMWInst::Xchg, Ptr, UndefValue::get(Ty), Alignment, Ordering, SSID); // Undef used as placeholder
    RMW->copyMetadata(Modify);
    Builder.SetInsertPoint(&Modify);
    ModifyOp->setArgOperand(LoadedOp, RMW);
    for (int attempts = 0; ; ) {
      FreezeInst *TrackReturn = Builder.Insert(new FreezeInst(ModifyOp)); // Create a temporary TrackingVH so we can recover the NewVal after inlining
      InlineFunctionInfo IFI;
      if (!InlineFunction(*ModifyOp, IFI).isSuccess()) {
        // Undo the attempt, since inlining failed
        BinOp = AtomicRMWInst::BAD_BINOP;
        TrackReturn->eraseFromParent();
        break;
      }
      ModifyOp = nullptr;
      NewVal = TrackReturn->getOperand(0);
      TrackReturn->eraseFromParent();
      // NewVal might have been folded away by inlining so redo patternMatchAtomicRMWOp here
      // tracing from RMW to NewVal, in case instsimplify folded something
      BinOp = patternMatchAtomicRMWOp(RMW, ValOp, tryinline, NewVal);
      if (tryinline) {
        ModifyOp = cast<CallInst>(NewVal);
        LoadedOp = ValOp;
        assert(ModifyOp->getArgOperand(LoadedOp) == RMW);
        ModifyOp->moveAfter(RMW); // NewValInst is a user of RMW, and has no other dependants (per patternMatchAtomicRMWOp)
        if (++attempts > 2)
          break;
        if (auto FOp = ModifyOp->getCalledFunction())
          BinOp = patternMatchAtomicRMWOp(FOp->getArg(LoadedOp), ValOp, tryinline, nullptr);
        else
          break;
        if (!tryinline && BinOp == AtomicRMWInst::BAD_BINOP)
          break;
      } else {
        assert(BinOp != AtomicRMWInst::BAD_BINOP);
        assert(isa<UndefValue>(RMW->getOperand(1))); // RMW was previously being used as the placeholder for Val
        Value *Val;
        if (ValOp != -1u) {
          Instruction *NewValInst = cast<Instruction>(NewVal);
          NewValInst->moveAfter(RMW); // NewValInst is a user of RMW, and has no other dependants (per patternMatchAtomicRMWOp)
          Val = NewValInst->getOperand(ValOp);
        } else if (BinOp == AtomicRMWInst::Xchg) {
          Val = NewVal;
        } else {
          // convert to an atomic fence of the form: atomicrmw or %ptr, 0
          assert(BinOp == AtomicRMWInst::Or);
          Val = ConstantInt::getNullValue(Ty);
        }
        RMW->setOperation(BinOp);
        RMW->setOperand(1, Val);
        OldVal = RMW;
        break;
      }
    }
    if (BinOp == AtomicRMWInst::BAD_BINOP) {
      ModifyOp->setArgOperand(LoadedOp, UndefValue::get(Ty));
      RMW->eraseFromParent();
    }
  }

  if (BinOp == AtomicRMWInst::BAD_BINOP) {
    // FIXME: If FP exceptions are observable, we should force them off for the
    // loop for the FP atomics.
    std::tie(OldVal, NewVal) = insertRMWCmpXchgLoop(
      Builder, Ty,  Ptr, *Alignment, Ordering, SSID, Modify,
      [&](IRBuilderBase &Builder, Value *Loaded) {
        ModifyOp->setArgOperand(LoadedOp, Loaded);
        ModifyOp->moveBefore(*Builder.GetInsertBlock(), Builder.GetInsertPoint());
        return ModifyOp;
      },
      CreateWeakCmpXchg);
  }

  for (auto user : make_early_inc_range(Modify.users())) {
    if (auto EV = dyn_cast<ExtractValueInst>(user)) {
      if (EV->getNumIndices() == 1) {
        if (EV->use_empty()) {
          EV->eraseFromParent();
          continue;
        }
        else if (EV->getIndices()[0] == 0) {
          EV->replaceAllUsesWith(OldVal);
          EV->eraseFromParent();
          continue;
        } else if (EV->getIndices()[0] == 1) {
          EV->replaceAllUsesWith(NewVal);
          EV->eraseFromParent();
          continue;
        }
      }
    }
  }
  if (!Modify.use_empty()) {
    auto OldNewVal = Builder.CreateInsertValue(UndefValue::get(Modify.getType()), OldVal, 0);
    OldNewVal = Builder.CreateInsertValue(OldNewVal, NewVal, 1);
    Modify.replaceAllUsesWith(OldNewVal);
  }
  Modify.eraseFromParent();
}

static bool expandAtomicModify(Function &F) {
  SmallVector<CallInst*> AtomicInsts;

  // Changing control-flow while iterating through it is a bad idea, so gather a
  // list of all atomic instructions before we start.
  for (Instruction &I : instructions(F))
    if (auto CI = dyn_cast<CallInst>(&I)) {
      auto callee = dyn_cast_or_null<Function>(CI->getCalledOperand());
      if (callee && callee->getName().starts_with("julia.atomicmodify.")) {
        assert(CI->getFunctionType() == callee->getFunctionType());
        AtomicInsts.push_back(CI);
      }
    }

  bool MadeChange = !AtomicInsts.empty();
  for (auto *I : AtomicInsts)
    expandAtomicModifyToCmpXchg(*I, createWeakCmpXchgInstFun);
  return MadeChange;
}

PreservedAnalyses ExpandAtomicModifyPass::run(Function &F, FunctionAnalysisManager &AM)
{
    if (expandAtomicModify(F)) {
        return PreservedAnalyses::none();
    }
    return PreservedAnalyses::all();
}
