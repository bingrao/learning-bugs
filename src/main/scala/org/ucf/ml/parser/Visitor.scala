package org.ucf.ml
package parser

import com.github.javaparser.ast.body.MethodDeclaration
import com.github.javaparser.ast.visitor.VoidVisitorAdapter

import scala.collection.mutable.ListBuffer

trait Visitor {
  class MethodNameCollector extends VoidVisitorAdapter[ListBuffer[String]] {
    override def visit(md:MethodDeclaration, c:ListBuffer[String]): Unit = {
      super.visit(md,c)
      c.+=(md.getNameAsString)
    }
  }
}
