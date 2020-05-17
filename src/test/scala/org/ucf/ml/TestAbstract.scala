package org.ucf.ml
/**
 * @author
 */
import org.junit.Test
import org.junit.Assert._

class TestAbstract extends TestUtils {
  @Test def testAbstract() {
    val inputClass =
      """
        |package org.ucf.ml;
        |
        |import java.util.Arrays;
        |import java.util.List;
        |import org.apache.commons.lang3.StringUtils;
        |
        |public class JavaApp {
        |
        |    public void hello(String input) {
        |        List<String> messages = Arrays.asList("hello", "baeldung", "readers!");
        |        messages.forEach(word -> StringUtils.capitalize(word));
        |        messages.forEach(StringUtils::capitalize);
        |
        |        List<Integer> numbers = Arrays.asList(5, 3, 50, 24, 40, 2, 9, 18);
        |        numbers.stream().sorted((a, b) -> a.compareTo(b));
        |        numbers.stream().sorted(Integer::compareTo);
        |    }
        |}
        |""".stripMargin
    get_abstract_code(inputClass, CLASS, false)
  }

  @Test def testAbstractFile_68(): Unit ={
    val input = "src/data/raw/fixed/1162.java"
    get_abstract_code(input, METHOD, true)
  }

  @Test def testAbstractFile(): Unit ={
    val input = "src/data/1/buggy.java"
    get_abstract_code(input, METHOD, true)
  }

  @Test def testPairAbstract():Unit = {
    val file_index = 1161
    val buggy = s"src/data/raw/buggy/${file_index}.java"
    val fixed = s"src/data/raw/fixed/${file_index}.java"
    single_task(buggy, fixed)
  }
}
