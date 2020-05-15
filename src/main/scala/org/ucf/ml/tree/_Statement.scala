package org.ucf.ml
package tree

import com.github.javaparser.ast.stmt._
import com.github.javaparser.ast.body.VariableDeclarator
import com.github.javaparser.ast.expr._
import org.ucf.ml.Context

import scala.collection.JavaConversions._


trait _Statement extends _Node {

  /************************** Statement ***************************/

  implicit class genStatement(node:Statement) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case n:ForEachStmt => n.genCode(ctx, numsIntent)
        case n:LocalClassDeclarationStmt => n.genCode(ctx, numsIntent)
        case n:ContinueStmt => n.genCode(ctx, numsIntent)
        case n:ExpressionStmt => n.genCode(ctx, numsIntent)
        case n:LabeledStmt => n.genCode(ctx, numsIntent)
        case n:YieldStmt => n.genCode(ctx, numsIntent)
        case n:ReturnStmt => n.genCode(ctx, numsIntent)
        case n:WhileStmt => n.genCode(ctx, numsIntent)
        case n:EmptyStmt => n.genCode(ctx, numsIntent)
        case n:UnparsableStmt => n.genCode(ctx, numsIntent)
        case n:IfStmt => n.genCode(ctx, numsIntent)
        case n:BreakStmt => n.genCode(ctx, numsIntent)
        case n:AssertStmt => n.genCode(ctx, numsIntent)
        case n:ExplicitConstructorInvocationStmt => n.genCode(ctx, numsIntent)
        case n:DoStmt => n.genCode(ctx, numsIntent)
        case n:ForStmt => n.genCode(ctx, numsIntent)
        case n:ThrowStmt => n.genCode(ctx, numsIntent)
        case n:TryStmt => n.genCode(ctx, numsIntent)
        case n:SwitchStmt => n.genCode(ctx, numsIntent)
        case n:SynchronizedStmt => n.genCode(ctx, numsIntent)
        case n:BlockStmt => n.genCode(ctx, numsIntent)
      }
      
    }
  }

  implicit class genForEachStmt(node:ForEachStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val integral = node.getIterable
      val variable = node.getVariable
      val body = node.getBody

      ctx.append("for")
      ctx.append("(")
      variable.genCode(ctx, numsIntent)
      ctx.append(":")
      integral.genCode(ctx, numsIntent)
      ctx.append(")")
      body.genCode(ctx, numsIntent)
    }
  }

  implicit class genLocalClassDeclarationStmt(node:LocalClassDeclarationStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO not for this project
    }
  }

  implicit class genContinueStmt(node:ContinueStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label =  node.getLabel
      ctx.append("continue")
      if (label.isPresent) label.get().genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genExpressionStmt(node:ExpressionStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      node.getExpression.genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genLabeledStmt(node:LabeledStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label = node.getLabel
      val sts = node.getStatement

      label.genCode(ctx, numsIntent)
      ctx.append(":")
      sts.genCode(ctx, numsIntent)

      ctx.append(";")
      ctx.appendNewLine()

    }
  }

  implicit class genYieldStmt(node:YieldStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append("yield")
      node.getExpression.genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genReturnStmt(node:ReturnStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append("return")
      val expr = node.getExpression
      if (expr.isPresent) expr.get().genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genWhileStmt(node:WhileStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val body = node.getBody
      val condition = node.getCondition
      ctx.append("while")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")
      ctx.appendNewLine()
      body.genCode(ctx, numsIntent)
    }
  }

  implicit class genEmptyStmt(node:EmptyStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genUnparsableStmt(node:UnparsableStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO, not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genIfStmt(node:IfStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val condition = node.getCondition
      val thenStmt = node.getThenStmt
      val elseStmt = node.getElseStmt

      ctx.append("if")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")

      ctx.appendNewLine()
      thenStmt.genCode(ctx, numsIntent)

      if (elseStmt.isPresent){
        ctx.append("else")
        ctx.appendNewLine()
        elseStmt.get().genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genBreakStmt(node:BreakStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label = node.getLabel
      ctx.append("break")
      if (label.isPresent) label.get().genCode(ctx, numsIntent)

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genAssertStmt(node:AssertStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val check = node.getCheck
      val msg = node.getMessage
      ctx.append("assert")
      check.genCode(ctx, numsIntent)

      if (msg.isPresent) {
        ctx.append(":")
        msg.get().genCode(ctx, numsIntent)
      }

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genExplicitConstructorInvocationStmt(node:ExplicitConstructorInvocationStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO, not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genDoStmt(node:DoStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val body = node.getBody
      val condition = node.getCondition

      body.genCode(ctx, numsIntent)
      ctx.append("while")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genForStmt(node:ForStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append("for")
      ctx.append("(")
      val initial = node.getInitialization.toList
      initial.foreach(init => {
        init.genCode(ctx, numsIntent)
        if (init != initial.last) ctx.append(",")
      })
      ctx.append(";")

      val compare = node.getCompare
      if (compare.isPresent) compare.get().genCode(ctx, numsIntent)

      ctx.append(";")

      val update = node.getUpdate.toList
      update.foreach(up => {
        up.genCode(ctx, numsIntent)
        if (up != update.last) ctx.append(",")
      })
      ctx.append(")")

      val body = node.getBody
      body.genCode(ctx, numsIntent)
    }
  }

  implicit class genThrowStmt(node:ThrowStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genTryStmt(node:TryStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genSwitchStmt(node:SwitchStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO, need more work here in future
      val entries = node.getEntries.toList
      val selector = node.getSelector

      ctx.append("switch")
      ctx.append("(")
      selector.genCode(ctx, numsIntent)
      ctx.append(")")
      ctx.appendNewLine()

      ctx.append("{")

      entries.foreach(entry => {
//        ctx.append("case")
//        ctx.append(":")
//        ctx.appendNewLine()
        ctx.append(entry.toString)
      })

      ctx.append("}")
    }
  }

  implicit class genSynchronizedStmt(node:SynchronizedStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genBlockStmt(node:BlockStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append("{")
      ctx.appendNewLine()
      node.getStatements.toList.foreach(sts => sts.genCode(ctx, numsIntent))
      ctx.append("}")
      ctx.appendNewLine()
    }
  }


  /******************************* Expression ********************************/
  implicit class genExpression(node:Expression){
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr:ArrayAccessExpr  => expr.genCode(ctx, numsIntent)
        case expr:ClassExpr  => expr.genCode(ctx, numsIntent)
        case expr:LambdaExpr  => expr.genCode(ctx, numsIntent)
        case expr:ArrayCreationExpr  => expr.genCode(ctx, numsIntent)
        case expr:ConditionalExpr  => expr.genCode(ctx, numsIntent)
        case expr:MethodCallExpr  => expr.genCode(ctx, numsIntent)
        case expr:AnnotationExpr  => expr.genCode(ctx, numsIntent)
        case expr:AssignExpr  => expr.genCode(ctx, numsIntent)
        case expr:InstanceOfExpr  => expr.genCode(ctx, numsIntent)
        case expr:ThisExpr  => expr.genCode(ctx, numsIntent)
        case expr:NameExpr  => expr.genCode(ctx, numsIntent)
        case expr:CastExpr  => expr.genCode(ctx, numsIntent)
        case expr:MethodReferenceExpr  => expr.genCode(ctx, numsIntent)
        case expr:EnclosedExpr  => expr.genCode(ctx, numsIntent)
        case expr:VariableDeclarationExpr  => expr.genCode(ctx, numsIntent)
        case expr:SwitchExpr  => expr.genCode(ctx, numsIntent)
        case expr:LiteralExpr => expr.genCode(ctx, numsIntent)
        case expr:ObjectCreationExpr  => expr.genCode(ctx, numsIntent)
        case expr:SuperExpr  => expr.genCode(ctx, numsIntent)
        case expr:UnaryExpr  => expr.genCode(ctx, numsIntent)
        case expr:BinaryExpr  => expr.genCode(ctx, numsIntent)
        case expr:FieldAccessExpr  => expr.genCode(ctx, numsIntent)
        case expr:TypeExpr  => expr.genCode(ctx, numsIntent)
        case expr:ArrayInitializerExpr  => expr.genCode(ctx, numsIntent)
      }
    }
  }


  implicit class genArrayAccessExpr(node:ArrayAccessExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val name = node.getName
      val index = node.getIndex
      name.genCode(ctx, numsIntent)
      ctx.append("[")
      index.genCode(ctx, numsIntent)
      ctx.append("]")
    }
  }


  implicit class genClassExpr(node:ClassExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {
      //TODO, not for this problems
      ctx.append(node.toString)
    }
  }


  implicit class genLambdaExpr(node:LambdaExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val parameters = node.getParameters.toList

      if (parameters.size > 1)
        ctx.append("(")

      parameters.foreach(p => {
        p.genCode(ctx, numsIntent)
        if (p != parameters.last) ctx.append(",")
      })

      if (parameters.size > 1)
        ctx.append(")")

      ctx.append("->")

      val body = node.getBody

      body.genCode(ctx, numsIntent)

    }
  }


  implicit class genArrayCreationExpr(node:ArrayCreationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val eleType = node.getElementType
      val initial = node.getInitializer
      val levels = node.getLevels.toList

      ctx.append("new")
      eleType.genCode(ctx, numsIntent)
      for (level <- levels){
        ctx.append("[")
        val dim = level.getDimension
        if (dim.isPresent) dim.get().genCode(ctx, numsIntent)
        ctx.append("]")
      }

      if (initial.isPresent) initial.get().genCode(ctx, numsIntent)
    }
  }

  implicit class genConditionalExpr(node:ConditionalExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val condition = node.getCondition
      val thenExpr = node.getThenExpr
      val elseExpr = node.getElseExpr

      condition.genCode(ctx, numsIntent)
      ctx.append("?")
      thenExpr.genCode(ctx, numsIntent)
      ctx.append(":")
      elseExpr.genCode(ctx, numsIntent)
    }
  }

  implicit class genMethodCallExpr(node:MethodCallExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val scope = node.getScope
      val arguments = node.getArguments.toList

      if (scope.isPresent) {
        //        val scope_value = ctx.variable_maps.getNewContent(scope.get().toString)
        //        ctx.append(scope_value)
        scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }

      val funcName = ctx.method_maps.getNewContent(node.getName.asString())
      ctx.append(funcName)

      ctx.append("(")
      arguments.foreach(expr => {
        expr.genCode(ctx, numsIntent)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")
    }
  }

  implicit class genAnnotationExpr(node:AnnotationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, not for this projects
      ctx.append(node.toString)
    }
  }

  implicit class genAssignExpr(node:AssignExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {
      val left = node.getTarget
      val right = node.getValue
      val op = node.getOperator

      left.genCode(ctx, numsIntent)

      ctx.append(op.asString())

      right.genCode(ctx, numsIntent)


    }
  }

  implicit class genInstanceOfExpr(node:InstanceOfExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, Not for this project
      //      pobj instanceof Child
      ctx.append(node.toString)
    }
  }

  implicit class genThisExpr(node:ThisExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append("This")
    }
  }

  implicit class genNameExpr(node:NameExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //      node.getName.genCode(ctx, numsIntent)

      ctx.append(ctx.variable_maps.getNewContent(node.getName.asString()))

    }
  }

  implicit class genCastExpr(node:CastExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO no implement for this project
      ctx.append(node.toString)
    }
  }

  implicit class genMethodReferenceExpr(node:MethodReferenceExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val ident = node.getIdentifier
      val scope = node.getScope

      scope.genCode(ctx, numsIntent)
      ctx.append("::")
      ctx.append(ident)
    }
  }

  implicit class genEnclosedExpr(node:EnclosedExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {
      //TODO not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genVariableDeclarationExpr(node:VariableDeclarationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node.getModifiers.toList.foreach(_.genCode(ctx, numsIntent))
      val varibles = node.getVariables.toList
      varibles.foreach(_.genCode(ctx, numsIntent))
    }
  }

  implicit class genSwitchExpr(node:SwitchExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, not for this project
      ctx.append(node.toString)
    }
  }

  // subclass
  implicit class genLiteralExpr(node:LiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr:NullLiteralExpr => ctx.append(expr.toString)
        case expr:BooleanLiteralExpr => ctx.append(expr.getValue.toString)
        case expr:LiteralStringValueExpr  => expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genLiteralStringValueExpr(node:LiteralStringValueExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr: TextBlockLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: CharLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: DoubleLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: LongLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: StringLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: IntegerLiteralExpr => expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genTextBlockLiteralExpr(node:TextBlockLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.textBlock_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  implicit class genCharLiteralExpr(node:CharLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.char_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  implicit class genDoubleLiteralExpr(node:DoubleLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.double_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  implicit class genLongLiteralExpr(node:LongLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.long_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  implicit class genStringLiteralExpr(node:StringLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.string_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  implicit class genIntegerLiteralExpr(node:IntegerLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = ctx.int_maps.getNewContent(node.getValue)
      ctx.append(value)
    }
  }

  /**
   *  new B().new C();
   *  scope --> new B()
   *  type --> new C()
   * @param node
   */
  implicit class genObjectCreationExpr(node:ObjectCreationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val arguments = node.getArguments.toList
      val scope = node.getScope
      val tp = node.getType

      if (scope.isPresent) {
        scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }

      ctx.append("new")

      tp.genCode(ctx, numsIntent)

      ctx.append("(")
      arguments.foreach(expr =>{
        expr.genCode(ctx, numsIntent)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")
    }
  }


  implicit class genSuperExpr(node:SuperExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, not for this project
      ctx.append(node.toString)
    }
  }

  implicit class genUnaryExpr(node:UnaryExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val op = node.getOperator.asString()
      val expr = node.getExpression
      if (node.isPostfix) {
        expr.genCode(ctx, numsIntent)
        ctx.append(op)
      }
      if (node.isPrefix) {
        ctx.append(op)
        expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genBinaryExpr(node:BinaryExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val left = node.getLeft
      val op = node.getOperator.asString()
      val right = node.getRight
      left.genCode(ctx, numsIntent)
      ctx.append(op)
      right.genCode(ctx, numsIntent)
    }
  }

  implicit class genFieldAccessExpr(node:FieldAccessExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
//      val scope_value = ctx.variable_maps.getNewContent(node.getScope.toString)
//      ctx.append(scope_value)
      node.getScope.genCode(ctx, numsIntent)

      ctx.append(".")

      // filed
      val name = ctx.variable_maps.getNewContent(node.getName.asString())
      ctx.append(name)



    }
  }

  implicit class genTypeExpr(node:TypeExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node.getType.genCode(ctx, numsIntent)
    }
  }

  implicit class genArrayInitializerExpr(node:ArrayInitializerExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val values = node.getValues.toList
      ctx.append("{")
      values.foreach(ele => {
        ele.genCode(ctx, numsIntent)
        if (ele != values.last) ctx.append(",")
      })
      ctx.append("}")
    }
  }

  /******************************* Variable ********************************/
  implicit class genVariableDeclarator(node: VariableDeclarator) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val tp = node.getType
      val name = node.getName
      val init = node.getInitializer

      tp.genCode(ctx, numsIntent)

      ctx.append(ctx.variable_maps.getNewContent(name.asString()))

      if (init.isPresent){
        ctx.append("=")
        init.get().genCode(ctx, numsIntent)
      }
    }
  }

}
