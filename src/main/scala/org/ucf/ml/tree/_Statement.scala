package org.ucf.ml
package tree

import scala.collection.JavaConversions._
import com.github.javaparser.ast.stmt._
import utils.{Context}



trait _Statement extends _Expression {

  /************************** Statement ***************************/
  implicit class genStatement(node:Statement) {
    def genCode(ctx:Context):String = {
      node match {
        case n:ForEachStmt => n.genCode(ctx)
        case n:LocalClassDeclarationStmt => n.genCode(ctx)
        case n:ContinueStmt => n.genCode(ctx)
        case n:ExpressionStmt => n.genCode(ctx)
        case n:LabeledStmt => n.genCode(ctx)
        case n:YieldStmt => n.genCode(ctx)
        case n:ReturnStmt => n.genCode(ctx)
        case n:WhileStmt => n.genCode(ctx)
        case n:EmptyStmt => n.genCode(ctx)
        case n:UnparsableStmt => n.genCode(ctx)
        case n:IfStmt => n.genCode(ctx)
        case n:BreakStmt => n.genCode(ctx)
        case n:AssertStmt => n.genCode(ctx)
        case n:ExplicitConstructorInvocationStmt => n.genCode(ctx)
        case n:DoStmt => n.genCode(ctx)
        case n:ForStmt => n.genCode(ctx)
        case n:ThrowStmt => n.genCode(ctx)
        case n:TryStmt => n.genCode(ctx)
        case n:SwitchStmt => n.genCode(ctx)
        case n:SynchronizedStmt => n.genCode(ctx)
        case n:BlockStmt => n.genCode(ctx)
      }
      EMPTY_STRING
    }
  }

  implicit class genForEachStmt(node:ForEachStmt) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genLocalClassDeclarationStmt(node:LocalClassDeclarationStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genContinueStmt(node:ContinueStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genExpressionStmt(node:ExpressionStmt) {
    def genCode(ctx:Context):String  = {
      node.getExpression.genCode(ctx)
      EMPTY_STRING
    }
  }

  implicit class genLabeledStmt(node:LabeledStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genYieldStmt(node:YieldStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genReturnStmt(node:ReturnStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genWhileStmt(node:WhileStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genEmptyStmt(node:EmptyStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genUnparsableStmt(node:UnparsableStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genIfStmt(node:IfStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genBreakStmt(node:BreakStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genAssertStmt(node:AssertStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genExplicitConstructorInvocationStmt(node:ExplicitConstructorInvocationStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genDoStmt(node:DoStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genForStmt(node:ForStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genThrowStmt(node:ThrowStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genTryStmt(node:TryStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genSwitchStmt(node:SwitchStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genSynchronizedStmt(node:SynchronizedStmt) {
    def genCode(ctx:Context):String  = {
      EMPTY_STRING
    }
  }

  implicit class genBlockStmt(node:BlockStmt) {
    def genCode(ctx:Context):String  = {
      ctx.append("{")
      ctx.append("\n")
      node.getStatements.toList.foreach(sts => sts.genCode(ctx))
      ctx.append("}")
      ctx.append("\n")
      EMPTY_STRING
    }
  }

}
