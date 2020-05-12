package org.ucf.ml
package tree

import scala.collection.JavaConversions._
import com.github.javaparser.ast.Modifier
import com.github.javaparser.ast.`type`.Type
import com.github.javaparser.ast.body.{Parameter, VariableDeclarator}
import com.github.javaparser.ast.expr.SimpleName
import utils.{Common, Context}

trait _Share extends Common{

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
      ctx.append(node.asString())
      EMPTY_STRING
    }
  }

  implicit class genParameter(node:Parameter) {
    def genCode(ctx:Context):String = {
      ctx.append(node.toString)
      EMPTY_STRING
    }
  }

}
