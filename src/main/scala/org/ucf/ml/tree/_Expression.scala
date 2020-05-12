package org.ucf.ml
package tree

import com.github.javaparser.ast.body.{Parameter, VariableDeclarator}
import utils.{Common, Context}
import com.github.javaparser.ast.expr._

import scala.collection.JavaConversions._

trait _Expression extends _Share {

  /******************************* Expression ********************************/
  implicit class gen(node:Expression){
    def genCode(ctx:Context):String = {
      node match {
        case expr:ArrayAccessExpr  => expr.genCode(ctx)
        case expr:ClassExpr  => expr.genCode(ctx)
        case expr:LambdaExpr  => expr.genCode(ctx)
        case expr:ArrayCreationExpr  => expr.genCode(ctx)
        case expr:ConditionalExpr  => expr.genCode(ctx)
        case expr:MethodCallExpr  => expr.genCode(ctx)
        case expr:AnnotationExpr  => expr.genCode(ctx)
        case expr:AssignExpr  => expr.genCode(ctx)
        case expr:InstanceOfExpr  => expr.genCode(ctx)
        case expr:ThisExpr  => expr.genCode(ctx)
        case expr:NameExpr  => expr.genCode(ctx)
        case expr:CastExpr  => expr.genCode(ctx)
        case expr:MethodReferenceExpr  => expr.genCode(ctx)
        case expr:EnclosedExpr  => expr.genCode(ctx)
        case expr:VariableDeclarationExpr  => expr.genCode(ctx)
        case expr:SwitchExpr  => expr.genCode(ctx)


        case expr:LiteralExpr => expr.genCode(ctx)

        case expr:ObjectCreationExpr  => expr.genCode(ctx)
        case expr:SuperExpr  => expr.genCode(ctx)
        case expr:UnaryExpr  => expr.genCode(ctx)
        case expr:BinaryExpr  => expr.genCode(ctx)
        case expr:FieldAccessExpr  => expr.genCode(ctx)
        case expr:TypeExpr  => expr.genCode(ctx)
        case expr:ArrayInitializerExpr  => expr.genCode(ctx)
      }
      EMPTY_STRING
    }
  }


  implicit class genArrayAccessExpr(node:ArrayAccessExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genClassExpr(node:ClassExpr) {
    def genCode(ctx:Context):String= {
      EMPTY_STRING
    }
  }


  implicit class genLambdaExpr(node:LambdaExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.toString)
      EMPTY_STRING
    }
  }


  implicit class genArrayCreationExpr(node:ArrayCreationExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genConditionalExpr(node:ConditionalExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genMethodCallExpr(node:MethodCallExpr) {
    def genCode(ctx:Context):String = {

      val scope = node.getScope
      val arguments = node.getArguments.toList

      if (scope.isPresent) scope.get().genCode(ctx)

      ctx.append(".")

      node.getName.genCode(ctx)

      ctx.append("(")
      arguments.foreach(expr => {
        expr.genCode(ctx)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")

      if (node.getParentNode.isPresent){
        val pa = node.getParentNode.get()
        if (!pa.isInstanceOf[Expression]){
          ctx.append(";")
          ctx.append("\n")
        }
      }


      EMPTY_STRING
    }
  }


  implicit class genAnnotationExpr(node:AnnotationExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }



  implicit class genAssignExpr(node:AssignExpr) {
    def genCode(ctx:Context):String= {
      val left = node.getTarget
      val right = node.getValue
      val op = node.getOperator

      left.genCode(ctx)

      ctx.append(op.asString())

      right.genCode(ctx)

      ctx.append(";")
      ctx.append("\n")

      EMPTY_STRING
    }
  }


  implicit class genInstanceOfExpr(node:InstanceOfExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genThisExpr(node:ThisExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genNameExpr(node:NameExpr) {
    def genCode(ctx:Context):String = {
      node.getName.genCode(ctx)
      EMPTY_STRING
    }
  }

  implicit class genCastExpr(node:CastExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genMethodReferenceExpr(node:MethodReferenceExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }



  implicit class genEnclosedExpr(node:EnclosedExpr) {
    def genCode(ctx:Context):String= {
      EMPTY_STRING
    }
  }


  implicit class genVariableDeclarationExpr(node:VariableDeclarationExpr) {
    def genCode(ctx:Context):String = {
      node.getModifiers.toList.foreach(_.genCode(ctx))
      val varibles = node.getVariables.toList
      varibles.foreach(_.genCode(ctx))
      EMPTY_STRING
    }
  }


  implicit class genSwitchExpr(node:SwitchExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  // subclass
  implicit class genLiteralExpr(node:LiteralExpr) {
    def genCode(ctx:Context):String = {
      node match {
        case expr:NullLiteralExpr => {
          ctx.append("null")
        }
        case expr:BooleanLiteralExpr => {
          ctx.append(expr.getValue.toString)
        }
        case expr:LiteralStringValueExpr  => expr.genCode(ctx)
      }
      EMPTY_STRING
    }
  }

  implicit class genLiteralStringValueExpr(node:LiteralStringValueExpr) {
    def genCode(ctx:Context):String = {
      node match {
        case expr: CharLiteralExpr => {expr.genCode(ctx)}
        case expr: LongLiteralExpr => {expr.genCode(ctx)}
        case expr: DoubleLiteralExpr => {expr.genCode(ctx)}
        case expr: StringLiteralExpr => {expr.genCode(ctx)}
        case expr: IntegerLiteralExpr => {expr.genCode(ctx)}
        case expr: TextBlockLiteralExpr => {expr.genCode(ctx)}
      }
      EMPTY_STRING
    }
  }

  implicit class genCharLiteralExpr(node:CharLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }

  implicit class genLongLiteralExpr(node:LongLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }

  implicit class genDoubleLiteralExpr(node:DoubleLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }

  implicit class genStringLiteralExpr(node:StringLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }

  implicit class genIntegerLiteralExpr(node:IntegerLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }

  implicit class genTextBlockLiteralExpr(node:TextBlockLiteralExpr) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getValue)
      EMPTY_STRING
    }
  }



  implicit class genObjectCreationExpr(node:ObjectCreationExpr) {
    def genCode(ctx:Context):String = {
      val arguments = node.getArguments.toList
      val scope = node.getScope
      val tp = node.getType

      ctx.append("new")
      if (scope.isPresent) {
        scope.get().genCode(ctx)
        ctx.append(".")
      }

      tp.genCode(ctx)

      ctx.append("(")
      arguments.foreach(expr =>{
        expr.genCode(ctx)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")

      EMPTY_STRING
    }
  }


  implicit class genSuperExpr(node:SuperExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genUnaryExpr(node:UnaryExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genBinaryExpr(node:BinaryExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }

  implicit class genFieldAccessExpr(node:FieldAccessExpr) {
    def genCode(ctx:Context):String = {
      node.getScope.genCode(ctx)

      ctx.append(".")

      node.getName.genCode(ctx)

      EMPTY_STRING
    }
  }

  implicit class genTypeExpr(node:TypeExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }


  implicit class genArrayInitializerExpr(node:ArrayInitializerExpr) {
    def genCode(ctx:Context):String = {
      EMPTY_STRING
    }
  }



  implicit class genVariableDeclarator(node: VariableDeclarator) {
    def genCode(ctx:Context):String = {
      val tp = node.getType
      val name = node.getName
      val init = node.getInitializer

      tp.genCode(ctx)
      name.genCode(ctx)

      if (init.isPresent){
        ctx.append("=")
        init.get().genCode(ctx)
      }
      ctx.append(";")
      ctx.append("\n")

      EMPTY_STRING
    }
  }

}
