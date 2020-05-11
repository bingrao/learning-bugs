package org.ucf.ml
package parser

import com.github.javaparser.ast.body.{MethodDeclaration, TypeDeclaration}
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.ast.expr.{MethodCallExpr, MethodReferenceExpr}

import scala.collection.mutable.ListBuffer

trait Visitor extends utils.context{
  case class MethodNameCollector() extends VoidVisitorAdapter[ListBuffer[String]] {
    override def visit(md:MethodDeclaration, c:ListBuffer[String]): Unit = {
      c.+=(md.getNameAsString)
      super.visit(md,c)
    }
  }

  case class MethodDeclCollector() extends VoidVisitorAdapter[ListBuffer[MethodDeclaration]] {
    override def visit(md:MethodDeclaration, arg: ListBuffer[MethodDeclaration]): Unit = {
      arg.+=(md)
      super.visit(md,arg)
    }
  }

  case class MethodCallCollector() extends VoidVisitorAdapter[ListBuffer[MethodCallExpr]] {
    override def visit(n: MethodCallExpr, arg: ListBuffer[MethodCallExpr]): Unit = {
      arg.+=(n)
      super.visit(n, arg)
    }
  }

  case class MethodRefCollector() extends VoidVisitorAdapter[ListBuffer[MethodReferenceExpr]] {
    override def visit(n: MethodReferenceExpr, arg: ListBuffer[MethodReferenceExpr]): Unit = {
      arg.+=(n)
      super.visit(n, arg)
    }
  }

}
