package org.ucf.ml
package tree

import com.github.javaparser.ast._
import com.github.javaparser.ast.`type`._
import com.github.javaparser.ast.body.Parameter
import com.github.javaparser.ast.expr.{SimpleName, Name}
import utils.{Common, Context}
import scala.collection.JavaConversions._
trait _Node extends Common {

  implicit class addPosition(node:Node) {
    def getPosition(ctx: utils.Context) = ctx.getNewPosition
  }

  implicit class genSimpleName(node:SimpleName) {
    def genCode(ctx:Context):Unit = {
      ctx.append(node.getIdentifier)
      
    }
  }

  implicit class genModifier(node: Modifier) {
    def genCode(ctx:Context):Unit = {
      ctx.append(node.getKeyword.asString())
      
    }
  }

  implicit class genType(node:Type) {
    def genCode(ctx:Context):Unit = {

      node match {
        case tp:UnionType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:VarType  =>{
          val value = ctx.type_maps.getNewContent(node.asString())
          ctx.append(value)
        }
        case tp:ReferenceType  => tp.genCode(ctx)
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
    def genCode(ctx:Context):Unit = {
      node match {
        case tp: ArrayType => tp.genCode(ctx)
        case tp: TypeParameter => tp.genCode(ctx)
        case tp: ClassOrInterfaceType => tp.genCode(ctx)
      }
    }
  }

  /**
   * So, int[][] becomes ArrayType(ArrayType(int)).
   * @param node
   */
  implicit class genArrayType(node:ArrayType) {
    def genCode(ctx:Context):Unit = {
      //TODO, need more details about
      val origin = node.getOrigin
      val comType = node.getComponentType
      comType.genCode(ctx)
      ctx.append("[")
      ctx.append("]")
    }
  }

  implicit class genTypeParameter(node:TypeParameter) {
    def genCode(ctx:Context):Unit = {
      val name = node.getName
      val typeBound = node.getTypeBound.toList

      ctx.append("<")
      name.genCode(ctx)
      if (typeBound.size() != 0){
        ctx.append("extends")
        typeBound.foreach(bound => {
          bound.genCode(ctx)
          if (bound != typeBound.last) ctx.append("&")
        })
      }
      ctx.append(">")
    }
  }

  implicit class genClassOrInterfaceType(node:ClassOrInterfaceType) {
    def genCode(ctx:Context):Unit = {
      val scope = node.getScope
      val name = node.getName
      val tps = node.getTypeArguments

      if (scope.isPresent){
        scope.get().genCode(ctx)
        ctx.append(".")
      }
      name.genCode(ctx)

      if (tps.isPresent){
        ctx.append("<")
        tps.get().toList.foreach(_.genCode(ctx))
        ctx.append(">")
      }
    }
  }


  implicit class genParameter(node:Parameter) {
    def genCode(ctx:Context):Unit = {
      val tp = node.getType
      val name = node.getName
      tp.genCode(ctx)

      val value = ctx.variable_maps.getNewContent(name.asString())
      ctx.append(value)
//      name.genCode(ctx)
    }
  }

  implicit class genName(node:Name) {
    def genCode(ctx:Context):Unit = {

      val qualifier = node.getQualifier
      if (qualifier.isPresent){
        qualifier.get().genCode(ctx)
        ctx.append(".")
      }

      val name = node.getIdentifier
      ctx.append(name)

    }


  }

}
