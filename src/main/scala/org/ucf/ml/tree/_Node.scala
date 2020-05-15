package org.ucf.ml
package tree

import com.github.javaparser.ast._
import com.github.javaparser.ast.`type`._
import com.github.javaparser.ast.body.Parameter
import com.github.javaparser.ast.expr.{FieldAccessExpr, MethodCallExpr, Name, SimpleName}
import utils.Common

import scala.collection.JavaConversions._
trait _Node extends Common {

  implicit class addPosition(node:Node) {
    def getPosition(ctx:Context, numsIntent:Int=0) = ctx.getNewPosition
  }

  implicit class genSimpleName(node:SimpleName) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(node.getIdentifier)
      
    }
  }

  implicit class genModifier(node: Modifier) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(node.getKeyword.asString())
      
    }
  }

  implicit class genType(node:Type) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      node match {
        case tp:UnionType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:VarType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:ReferenceType  => tp.genCode(ctx, numsIntent)
        case tp:UnknownType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:PrimitiveType  =>{
//          val value = ctx.type_maps.getNewContent(node.asString())
//          ctx.append(value)
          ctx.append(tp.asString())
        }
        case tp:WildcardType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:VoidType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:IntersectionType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
      }
    }
  }

  implicit class genReferenceType(node:ReferenceType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case tp: ArrayType => tp.genCode(ctx, numsIntent)
        case tp: TypeParameter => tp.genCode(ctx, numsIntent)
        case tp: ClassOrInterfaceType => tp.genCode(ctx, numsIntent)
      }
    }
  }

  /**
   * So, int[][] becomes ArrayType(ArrayType(int)).
   * @param node
   */
  implicit class genArrayType(node:ArrayType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, need more details about
      val origin = node.getOrigin
      val comType = node.getComponentType
      comType.genCode(ctx, numsIntent)
      ctx.append("[")
      ctx.append("]")
    }
  }

  implicit class genTypeParameter(node:TypeParameter) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val name = node.getName
      val typeBound = node.getTypeBound.toList

      ctx.append("<")
      name.genCode(ctx, numsIntent)
      if (typeBound.size() != 0){
        ctx.append("extends")
        typeBound.foreach(bound => {
          bound.genCode(ctx, numsIntent)
          if (bound != typeBound.last) ctx.append("&")
        })
      }
      ctx.append(">")
    }
  }

  implicit class genClassOrInterfaceType(node:ClassOrInterfaceType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val scope = node.getScope
      val name = node.getName
      val tps = node.getTypeArguments

      if (scope.isPresent){
        scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }
      name.genCode(ctx, numsIntent)

      if (tps.isPresent){
        ctx.append("<")
        tps.get().toList.foreach(_.genCode(ctx, numsIntent))
        ctx.append(">")
      }
    }
  }


  implicit class genParameter(node:Parameter) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val tp = node.getType
      val name = node.getName
      tp.genCode(ctx, numsIntent)

      val value = ctx.variable_maps.getNewContent(name.asString())
      ctx.append(value)
//      name.genCode(ctx, numsIntent)
    }
  }

  implicit class genName(node:Name) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val qualifier = node.getQualifier
      if (qualifier.isPresent){
        qualifier.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }

      val name = node.getIdentifier
      ctx.append(name)

    }
  }

  def expand_scope(ctx:Context, scope:Node):Unit = {
    scope match {
      case expr: MethodCallExpr => {
        val expr_name = expr.getName
        val expr_scope = expr.getScope

        // method name
        ctx.method_maps.getNewContent(expr_name.asString())

        if (expr_scope.isPresent) expand_scope(ctx, expr_scope.get())
      }
      case tp:ClassOrInterfaceType => {
        val tp_name = tp.getName
        val tp_scope = tp.getScope
        if (tp_scope.isPresent) expand_scope(ctx, tp_scope.get())
      }
      case fd:FieldAccessExpr => {}
      case _ =>{}
    }
  }

}
