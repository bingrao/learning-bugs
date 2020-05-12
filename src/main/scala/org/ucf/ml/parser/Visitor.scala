package org.ucf.ml
package parser

import com.github.javaparser.ast.body.{ClassOrInterfaceDeclaration, MethodDeclaration}
import com.github.javaparser.ast.visitor.{TreeVisitor, VoidVisitorAdapter}

import scala.collection.mutable.ListBuffer
import com.github.javaparser.ast.{Node, PackageDeclaration}
import org.ucf.ml.tree.EnrichedTrees

trait Visitor extends utils.Common with EnrichedTrees {

  case class MethodNameCollector() extends VoidVisitorAdapter[ListBuffer[String]] {
    override def visit(md:MethodDeclaration, c:ListBuffer[String]): Unit = {
      c.+=(md.getNameAsString)
      super.visit(md,c)
    }
  }


  case class addPositionVisitor(ctx:utils.Context) extends TreeVisitor {
    override def process(node: Node): Unit = {
      node match {
        case p: PackageDeclaration => {
          logger.info(f"${p.getName} -> ${p.getPosition(ctx)}")
        }
        case c: ClassOrInterfaceDeclaration => {
          logger.info(f"${c.getName} -> ${c.getPosition(ctx)}")
        }
        case m: MethodDeclaration => {
          logger.info(f"${m.getName} -> ${m.getPosition(ctx)}")
        }
        case _ =>{}
      }
    }
  }

}
