package org.ucf.ml
package tree

import com.github.javaparser.ast.Modifier
import com.github.javaparser.ast.`type`.Type
import com.github.javaparser.ast.body.{Parameter}
import com.github.javaparser.ast.expr.SimpleName
import utils.{Common, Context}

trait _Share extends Common {

  implicit class genSimpleName(node:SimpleName) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getIdentifier)
      EMPTY_STRING
    }
  }

  implicit class genModifier(node: Modifier) {
    def genCode(ctx:Context):String = {
      ctx.append(node.getKeyword.asString())
      EMPTY_STRING
    }
  }

  implicit class genType(node:Type) {
    def genCode(ctx:Context):String = {
      val value = ctx.type_maps.getNewContent(node.asString())
      ctx.append(value)
      EMPTY_STRING
    }
  }

  implicit class genParameter(node:Parameter) {
    def genCode(ctx:Context):String = {
      val tp = node.getType
      val name = node.getName
      tp.genCode(ctx)

      val value = ctx.variable_maps.getNewContent(name.asString())
      ctx.append(value)
//      name.genCode(ctx)

      EMPTY_STRING
    }
  }

}
