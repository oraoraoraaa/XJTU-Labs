#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SRC 65536
#define MAX_NAME 64
#define MAX_QUADS 4096
#define MAX_SYMBOLS 512
#define MAX_ERRORS 256

typedef enum
{
  TOK_EOF = 0,
  TOK_ID,
  TOK_NUM,
  TOK_INT,
  TOK_FLOAT,
  TOK_VOID,
  TOK_IF,
  TOK_ELSE,
  TOK_WHILE,
  TOK_RETURN,
  TOK_PRINT,
  TOK_INPUT,
  TOK_LPA,
  TOK_RPA,
  TOK_LBR,
  TOK_RBR,
  TOK_LBK,
  TOK_RBK,
  TOK_COMMA,
  TOK_SEMI,
  TOK_ADD,
  TOK_SUB,
  TOK_MUL,
  TOK_DIV,
  TOK_ASSIGN,
  TOK_EQ,
  TOK_NE,
  TOK_LT,
  TOK_LE,
  TOK_GT,
  TOK_GE
} TokenType;

typedef struct
{
  TokenType type;
  char lexeme[MAX_NAME];
  int line;
  int col;
} Token;

typedef struct
{
  const char *src;
  size_t pos;
  int line;
  int col;
} Lexer;

typedef enum
{
  TYPE_UNKNOWN = 0,
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_VOID
} ValueType;

typedef struct
{
  char name[MAX_NAME];
  ValueType type;
  int scope;
  int is_array;
  int array_size;
} Symbol;

typedef struct
{
  char op[16];
  char arg1[MAX_NAME];
  char arg2[MAX_NAME];
  char res[MAX_NAME];
} Quad;

typedef struct
{
  char name[MAX_NAME];
  ValueType type;
  int is_lvalue;
} Expr;

static Lexer g_lex;
static Token g_cur;
static Quad g_quads[MAX_QUADS];
static int g_quad_count = 0;
static int g_temp_count = 0;
static Symbol g_symbols[MAX_SYMBOLS];
static int g_symbol_count = 0;
static int g_scope_level = 0;
static ValueType g_func_ret = TYPE_VOID;
static char g_errors[MAX_ERRORS][MAX_NAME * 2];
static int g_error_count = 0;

static void
record_error (const char *msg)
{
  if (g_error_count >= MAX_ERRORS)
    {
      return;
    }
  strncpy (g_errors[g_error_count], msg, sizeof (g_errors[0]) - 1);
  g_errors[g_error_count][sizeof (g_errors[0]) - 1] = '\0';
  g_error_count++;
}

static void
lexer_init (const char *src)
{
  g_lex.src = src;
  g_lex.pos = 0;
  g_lex.line = 1;
  g_lex.col = 1;
}

static char
lexer_peek (void)
{
  return g_lex.src[g_lex.pos];
}

static char
lexer_next (void)
{
  char c = g_lex.src[g_lex.pos];
  if (c == '\0')
    {
      return c;
    }
  g_lex.pos++;
  if (c == '\n')
    {
      g_lex.line++;
      g_lex.col = 1;
    }
  else
    {
      g_lex.col++;
    }
  return c;
}

static void
skip_whitespace (void)
{
  while (1)
    {
      char c = lexer_peek ();
      if (isspace ((unsigned char)c))
        {
          lexer_next ();
          continue;
        }
      if (c == '/' && g_lex.src[g_lex.pos + 1] == '/')
        {
          while (c != '\0' && c != '\n')
            {
              c = lexer_next ();
            }
          continue;
        }
      if (c == '/' && g_lex.src[g_lex.pos + 1] == '*')
        {
          lexer_next ();
          lexer_next ();
          while (1)
            {
              c = lexer_peek ();
              if (c == '\0')
                {
                  break;
                }
              if (c == '*' && g_lex.src[g_lex.pos + 1] == '/')
                {
                  lexer_next ();
                  lexer_next ();
                  break;
                }
              lexer_next ();
            }
          continue;
        }
      break;
    }
}

static int
is_ident_start (char c)
{
  return isalpha ((unsigned char)c) || c == '_';
}

static int
is_ident_char (char c)
{
  return isalnum ((unsigned char)c) || c == '_';
}

static Token
make_token (TokenType type, const char *lexeme, int line, int col)
{
  Token t;
  t.type = type;
  strncpy (t.lexeme, lexeme, MAX_NAME - 1);
  t.lexeme[MAX_NAME - 1] = '\0';
  t.line = line;
  t.col = col;
  return t;
}

static Token
next_token (void)
{
  skip_whitespace ();
  int line = g_lex.line;
  int col = g_lex.col;
  char c = lexer_peek ();
  if (c == '\0')
    {
      return make_token (TOK_EOF, "", line, col);
    }

  if (is_ident_start (c))
    {
      char buf[MAX_NAME];
      int n = 0;
      while (is_ident_char (lexer_peek ()))
        {
          if (n < MAX_NAME - 1)
            {
              buf[n++] = lexer_next ();
            }
          else
            {
              lexer_next ();
            }
        }
      buf[n] = '\0';
      if (strcmp (buf, "int") == 0)
        {
          return make_token (TOK_INT, buf, line, col);
        }
      if (strcmp (buf, "float") == 0)
        {
          return make_token (TOK_FLOAT, buf, line, col);
        }
      if (strcmp (buf, "void") == 0)
        {
          return make_token (TOK_VOID, buf, line, col);
        }
      if (strcmp (buf, "if") == 0)
        {
          return make_token (TOK_IF, buf, line, col);
        }
      if (strcmp (buf, "else") == 0)
        {
          return make_token (TOK_ELSE, buf, line, col);
        }
      if (strcmp (buf, "while") == 0)
        {
          return make_token (TOK_WHILE, buf, line, col);
        }
      if (strcmp (buf, "return") == 0)
        {
          return make_token (TOK_RETURN, buf, line, col);
        }
      if (strcmp (buf, "print") == 0)
        {
          return make_token (TOK_PRINT, buf, line, col);
        }
      if (strcmp (buf, "input") == 0)
        {
          return make_token (TOK_INPUT, buf, line, col);
        }
      return make_token (TOK_ID, buf, line, col);
    }

  if (isdigit ((unsigned char)c))
    {
      char buf[MAX_NAME];
      int n = 0;
      int has_dot = 0;
      while (isdigit ((unsigned char)lexer_peek ()) || lexer_peek () == '.')
        {
          if (lexer_peek () == '.')
            {
              if (has_dot)
                {
                  break;
                }
              has_dot = 1;
            }
          if (n < MAX_NAME - 1)
            {
              buf[n++] = lexer_next ();
            }
          else
            {
              lexer_next ();
            }
        }
      buf[n] = '\0';
      return make_token (TOK_NUM, buf, line, col);
    }

  lexer_next ();
  switch (c)
    {
    case '(':
      return make_token (TOK_LPA, "(", line, col);
    case ')':
      return make_token (TOK_RPA, ")", line, col);
    case '{':
      return make_token (TOK_LBR, "{", line, col);
    case '}':
      return make_token (TOK_RBR, "}", line, col);
    case '[':
      return make_token (TOK_LBK, "[", line, col);
    case ']':
      return make_token (TOK_RBK, "]", line, col);
    case ',':
      return make_token (TOK_COMMA, ",", line, col);
    case ';':
      return make_token (TOK_SEMI, ";", line, col);
    case '+':
      return make_token (TOK_ADD, "+", line, col);
    case '-':
      return make_token (TOK_SUB, "-", line, col);
    case '*':
      return make_token (TOK_MUL, "*", line, col);
    case '/':
      return make_token (TOK_DIV, "/", line, col);
    case '=':
      if (lexer_peek () == '=')
        {
          lexer_next ();
          return make_token (TOK_EQ, "==", line, col);
        }
      return make_token (TOK_ASSIGN, "=", line, col);
    case '!':
      if (lexer_peek () == '=')
        {
          lexer_next ();
          return make_token (TOK_NE, "!=", line, col);
        }
      break;
    case '<':
      if (lexer_peek () == '=')
        {
          lexer_next ();
          return make_token (TOK_LE, "<=", line, col);
        }
      return make_token (TOK_LT, "<", line, col);
    case '>':
      if (lexer_peek () == '=')
        {
          lexer_next ();
          return make_token (TOK_GE, ">=", line, col);
        }
      return make_token (TOK_GT, ">", line, col);
    default:
      break;
    }

  {
    char msg[MAX_NAME * 2];
    snprintf (msg, sizeof (msg), "Unexpected character '%c' at %d:%d", c, line,
              col);
    record_error (msg);
  }
  return make_token (TOK_EOF, "", line, col);
}

static void
advance_token (void)
{
  g_cur = next_token ();
}

static int
match (TokenType type)
{
  if (g_cur.type == type)
    {
      advance_token ();
      return 1;
    }
  return 0;
}

static void
expect (TokenType type, const char *msg)
{
  if (!match (type))
    {
      char err[MAX_NAME * 2];
      snprintf (err, sizeof (err), "%s at %d:%d", msg, g_cur.line, g_cur.col);
      record_error (err);
    }
}

static void
emit_quad (const char *op, const char *arg1, const char *arg2, const char *res)
{
  if (g_quad_count >= MAX_QUADS)
    {
      record_error ("Too many quadruples");
      return;
    }
  Quad *q = &g_quads[g_quad_count++];
  strncpy (q->op, op, sizeof (q->op) - 1);
  q->op[sizeof (q->op) - 1] = '\0';
  strncpy (q->arg1, arg1 ? arg1 : "", MAX_NAME - 1);
  q->arg1[MAX_NAME - 1] = '\0';
  strncpy (q->arg2, arg2 ? arg2 : "", MAX_NAME - 1);
  q->arg2[MAX_NAME - 1] = '\0';
  strncpy (q->res, res ? res : "", MAX_NAME - 1);
  q->res[MAX_NAME - 1] = '\0';
}

static int
current_quad_index (void)
{
  return g_quad_count + 1;
}

static void
patch_quad (int index, int target)
{
  if (index <= 0 || index > g_quad_count)
    {
      return;
    }
  char buf[MAX_NAME];
  snprintf (buf, sizeof (buf), "%d", target);
  strncpy (g_quads[index - 1].res, buf, MAX_NAME - 1);
  g_quads[index - 1].res[MAX_NAME - 1] = '\0';
}

static void
new_temp (char *out)
{
  snprintf (out, MAX_NAME, "T%d", ++g_temp_count);
}

static void
enter_scope (void)
{
  g_scope_level++;
}

static void
leave_scope (void)
{
  if (g_scope_level > 0)
    {
      g_scope_level--;
    }
}

static int
find_symbol (const char *name, ValueType *type)
{
  for (int i = g_symbol_count - 1; i >= 0; --i)
    {
      if (g_symbols[i].scope > g_scope_level)
        {
          continue;
        }
      if (strcmp (g_symbols[i].name, name) == 0)
        {
          if (type)
            {
              *type = g_symbols[i].type;
            }
          return 1;
        }
    }
  return 0;
}

static int
declare_symbol (const char *name, ValueType type, int is_array, int array_size)
{
  for (int i = 0; i < g_symbol_count; ++i)
    {
      if (g_symbols[i].scope == g_scope_level
          && strcmp (g_symbols[i].name, name) == 0)
        {
          return 0;
        }
    }
  if (g_symbol_count >= MAX_SYMBOLS)
    {
      return 0;
    }
  Symbol *sym = &g_symbols[g_symbol_count++];
  strncpy (sym->name, name, MAX_NAME - 1);
  sym->name[MAX_NAME - 1] = '\0';
  sym->type = type;
  sym->scope = g_scope_level;
  sym->is_array = is_array;
  sym->array_size = array_size;
  return 1;
}

static ValueType
type_merge (ValueType a, ValueType b)
{
  if (a == TYPE_FLOAT || b == TYPE_FLOAT)
    {
      return TYPE_FLOAT;
    }
  if (a == TYPE_INT && b == TYPE_INT)
    {
      return TYPE_INT;
    }
  return TYPE_UNKNOWN;
}

static Expr parse_expression (void);
static void parse_statement (void);
static void parse_block (void);

static Expr
make_expr (const char *name, ValueType type, int is_lvalue)
{
  Expr e;
  strncpy (e.name, name, MAX_NAME - 1);
  e.name[MAX_NAME - 1] = '\0';
  e.type = type;
  e.is_lvalue = is_lvalue;
  return e;
}

static Expr
parse_primary (void)
{
  if (g_cur.type == TOK_NUM)
    {
      Expr e = make_expr (g_cur.lexeme, TYPE_INT, 0);
      if (strchr (g_cur.lexeme, '.') != NULL)
        {
          e.type = TYPE_FLOAT;
        }
      advance_token ();
      return e;
    }

  if (g_cur.type == TOK_ID)
    {
      char name[MAX_NAME];
      strncpy (name, g_cur.lexeme, MAX_NAME - 1);
      name[MAX_NAME - 1] = '\0';
      advance_token ();

      if (match (TOK_LPA))
        {
          int arg_count = 0;
          if (g_cur.type != TOK_RPA)
            {
              while (1)
                {
                  Expr arg = parse_expression ();
                  emit_quad ("param", arg.name, "", "");
                  arg_count++;
                  if (match (TOK_COMMA))
                    {
                      if (g_cur.type == TOK_RPA)
                        {
                          record_error ("Trailing comma in call");
                          break;
                        }
                      continue;
                    }
                  break;
                }
            }
          expect (TOK_RPA, "Missing ) in call");
          char tmp[MAX_NAME];
          new_temp (tmp);
          char argc_buf[MAX_NAME];
          snprintf (argc_buf, sizeof (argc_buf), "%d", arg_count);
          emit_quad ("call", name, argc_buf, tmp);
          return make_expr (tmp, TYPE_UNKNOWN, 0);
        }

      char full[MAX_NAME];
      strncpy (full, name, MAX_NAME - 1);
      full[MAX_NAME - 1] = '\0';
      ValueType t = TYPE_UNKNOWN;
      if (!find_symbol (name, &t))
        {
          char msg[MAX_NAME * 2];
          snprintf (msg, sizeof (msg), "Undeclared identifier '%s'", name);
          record_error (msg);
        }

      while (match (TOK_LBK))
        {
          Expr idx = parse_expression ();
          expect (TOK_RBK, "Missing ]");
          char buf[MAX_NAME];
          snprintf (buf, sizeof (buf), "%s[%s]", full, idx.name);
          strncpy (full, buf, MAX_NAME - 1);
          full[MAX_NAME - 1] = '\0';
        }
      return make_expr (full, t, 1);
    }

  if (g_cur.type == TOK_INPUT)
    {
      advance_token ();
      expect (TOK_LPA, "Missing ( after input");
      expect (TOK_RPA, "Missing ) after input");
      char tmp[MAX_NAME];
      new_temp (tmp);
      emit_quad ("input", "", "", tmp);
      return make_expr (tmp, TYPE_INT, 0);
    }

  if (match (TOK_LPA))
    {
      Expr e = parse_expression ();
      expect (TOK_RPA, "Missing )");
      return e;
    }

  if (match (TOK_SUB))
    {
      Expr e = parse_primary ();
      char tmp[MAX_NAME];
      new_temp (tmp);
      emit_quad ("neg", e.name, "", tmp);
      return make_expr (tmp, e.type, 0);
    }

  record_error ("Invalid expression");
  return make_expr ("", TYPE_UNKNOWN, 0);
}

static Expr
parse_term (void)
{
  Expr left = parse_primary ();
  while (g_cur.type == TOK_MUL || g_cur.type == TOK_DIV)
    {
      TokenType op = g_cur.type;
      advance_token ();
      Expr right = parse_primary ();
      char tmp[MAX_NAME];
      new_temp (tmp);
      emit_quad (op == TOK_MUL ? "*" : "/", left.name, right.name, tmp);
      left = make_expr (tmp, type_merge (left.type, right.type), 0);
    }
  return left;
}

static Expr
parse_additive (void)
{
  Expr left = parse_term ();
  while (g_cur.type == TOK_ADD || g_cur.type == TOK_SUB)
    {
      TokenType op = g_cur.type;
      advance_token ();
      Expr right = parse_term ();
      char tmp[MAX_NAME];
      new_temp (tmp);
      emit_quad (op == TOK_ADD ? "+" : "-", left.name, right.name, tmp);
      left = make_expr (tmp, type_merge (left.type, right.type), 0);
    }
  return left;
}

static Expr
parse_assignment_expr (void)
{
  Expr left = parse_additive ();
  if (g_cur.type == TOK_ASSIGN)
    {
      if (!left.is_lvalue)
        {
          record_error ("Left side of assignment is not assignable");
        }
      advance_token ();
      Expr right = parse_assignment_expr ();
      emit_quad ("=", right.name, "", left.name);
      left.is_lvalue = 0;
      left.type = right.type;
    }
  return left;
}

static Expr
parse_expression (void)
{
  return parse_assignment_expr ();
}

static void
parse_condition (char *op_out, Expr *left_out, Expr *right_out)
{
  Expr left = parse_expression ();
  if (g_cur.type == TOK_LT || g_cur.type == TOK_LE || g_cur.type == TOK_GT
      || g_cur.type == TOK_GE || g_cur.type == TOK_EQ || g_cur.type == TOK_NE)
    {
      TokenType op = g_cur.type;
      advance_token ();
      Expr right = parse_expression ();
      const char *op_str = "";
      switch (op)
        {
        case TOK_LT:
          op_str = "j<";
          break;
        case TOK_LE:
          op_str = "j<=";
          break;
        case TOK_GT:
          op_str = "j>";
          break;
        case TOK_GE:
          op_str = "j>=";
          break;
        case TOK_EQ:
          op_str = "j==";
          break;
        case TOK_NE:
          op_str = "j!=";
          break;
        default:
          break;
        }
      strncpy (op_out, op_str, 8);
      *left_out = left;
      *right_out = right;
      return;
    }

  strncpy (op_out, "j!=", 8);
  *left_out = left;
  *right_out = make_expr ("0", TYPE_INT, 0);
}

static void
parse_decl (ValueType type)
{
  while (1)
    {
      if (g_cur.type != TOK_ID)
        {
          record_error ("Expected identifier in declaration");
          return;
        }
      char name[MAX_NAME];
      strncpy (name, g_cur.lexeme, MAX_NAME - 1);
      name[MAX_NAME - 1] = '\0';
      advance_token ();

      int is_array = 0;
      int array_size = 0;
      if (match (TOK_LBK))
        {
          if (g_cur.type == TOK_NUM)
            {
              array_size = atoi (g_cur.lexeme);
              advance_token ();
            }
          else
            {
              record_error ("Array size must be number");
            }
          expect (TOK_RBK, "Missing ] in array");
          is_array = 1;
        }

      if (!declare_symbol (name, type, is_array, array_size))
        {
          char msg[MAX_NAME * 2];
          snprintf (msg, sizeof (msg), "Duplicate declaration of '%s'", name);
          record_error (msg);
        }

      if (match (TOK_ASSIGN))
        {
          Expr rhs = parse_expression ();
          emit_quad ("=", rhs.name, "", name);
        }

      if (match (TOK_COMMA))
        {
          continue;
        }
      break;
    }
}

static Expr
parse_lvalue (void)
{
  if (g_cur.type != TOK_ID)
    {
      record_error ("Expected identifier");
      return make_expr ("", TYPE_UNKNOWN, 0);
    }
  char name[MAX_NAME];
  strncpy (name, g_cur.lexeme, MAX_NAME - 1);
  name[MAX_NAME - 1] = '\0';
  advance_token ();

  ValueType t = TYPE_UNKNOWN;
  if (!find_symbol (name, &t))
    {
      char msg[MAX_NAME * 2];
      snprintf (msg, sizeof (msg), "Undeclared identifier '%s'", name);
      record_error (msg);
    }

  char full[MAX_NAME];
  strncpy (full, name, MAX_NAME - 1);
  full[MAX_NAME - 1] = '\0';
  while (match (TOK_LBK))
    {
      Expr idx = parse_expression ();
      expect (TOK_RBK, "Missing ] in index");
      char buf[MAX_NAME];
      snprintf (buf, sizeof (buf), "%s[%s]", full, idx.name);
      strncpy (full, buf, MAX_NAME - 1);
      full[MAX_NAME - 1] = '\0';
    }

  return make_expr (full, t, 1);
}

static void
parse_if (void)
{
  expect (TOK_IF, "Expected if");
  expect (TOK_LPA, "Missing ( after if");
  char op[8] = "";
  Expr left, right;
  parse_condition (op, &left, &right);
  expect (TOK_RPA, "Missing ) after if condition");

  int true_jump = current_quad_index ();
  emit_quad (op, left.name, right.name, "");
  int false_jump = current_quad_index ();
  emit_quad ("j", "", "", "");

  parse_statement ();
  if (g_cur.type == TOK_ELSE)
    {
      int end_jump = current_quad_index ();
      emit_quad ("j", "", "", "");
      patch_quad (true_jump, true_jump + 2);
      patch_quad (false_jump, current_quad_index ());
      advance_token ();
      parse_statement ();
      patch_quad (end_jump, current_quad_index ());
    }
  else
    {
      patch_quad (true_jump, true_jump + 2);
      patch_quad (false_jump, current_quad_index ());
    }
}

static void
parse_while (void)
{
  expect (TOK_WHILE, "Expected while");
  int loop_start = current_quad_index ();
  expect (TOK_LPA, "Missing ( after while");
  char op[8] = "";
  Expr left, right;
  parse_condition (op, &left, &right);
  expect (TOK_RPA, "Missing ) after while condition");

  int true_jump = current_quad_index ();
  emit_quad (op, left.name, right.name, "");
  int false_jump = current_quad_index ();
  emit_quad ("j", "", "", "");

  parse_statement ();
  emit_quad ("j", "", "", "");
  patch_quad (g_quad_count, loop_start);
  patch_quad (true_jump, true_jump + 2);
  patch_quad (false_jump, current_quad_index ());
}

static void
parse_return (void)
{
  expect (TOK_RETURN, "Expected return");
  if (g_cur.type != TOK_SEMI && g_cur.type != TOK_RBR)
    {
      Expr e = parse_expression ();
      if (g_func_ret == TYPE_VOID)
        {
          record_error ("Void function should not return a value");
        }
      else if (g_func_ret == TYPE_INT && e.type == TYPE_FLOAT)
        {
          record_error ("Return type mismatch (float to int)");
        }
      emit_quad ("ret", e.name, "", "");
    }
  else
    {
      if (g_func_ret != TYPE_VOID)
        {
          record_error ("Non-void function should return a value");
        }
      emit_quad ("ret", "", "", "");
    }
  match (TOK_SEMI);
}

static void
parse_print (void)
{
  expect (TOK_PRINT, "Expected print");
  if (match (TOK_LPA))
    {
      Expr e = parse_expression ();
      expect (TOK_RPA, "Missing ) after print");
      emit_quad ("print", e.name, "", "");
    }
  else
    {
      Expr e = parse_expression ();
      emit_quad ("print", e.name, "", "");
    }
  match (TOK_SEMI);
}

static void
parse_statement (void)
{
  if (g_cur.type == TOK_LBR)
    {
      parse_block ();
      return;
    }
  if (g_cur.type == TOK_IF)
    {
      parse_if ();
      return;
    }
  if (g_cur.type == TOK_WHILE)
    {
      parse_while ();
      return;
    }
  if (g_cur.type == TOK_RETURN)
    {
      parse_return ();
      return;
    }
  if (g_cur.type == TOK_PRINT)
    {
      parse_print ();
      return;
    }
  if (g_cur.type == TOK_INT || g_cur.type == TOK_FLOAT)
    {
      ValueType t = g_cur.type == TOK_INT ? TYPE_INT : TYPE_FLOAT;
      advance_token ();
      parse_decl (t);
      match (TOK_SEMI);
      return;
    }
  if (g_cur.type == TOK_ID)
    {
      Token lookahead = g_cur;
      advance_token ();
      if (g_cur.type == TOK_LPA)
        {
          g_cur = lookahead;
          Expr call = parse_expression ();
          (void)call;
          match (TOK_SEMI);
          return;
        }
      if (g_cur.type == TOK_ASSIGN || g_cur.type == TOK_LBK)
        {
          g_cur = lookahead;
          Expr lhs = parse_lvalue ();
          expect (TOK_ASSIGN, "Missing = in assignment");
          Expr rhs = parse_expression ();
          emit_quad ("=", rhs.name, "", lhs.name);
          match (TOK_SEMI);
          return;
        }
      g_cur = lookahead;
    }

  Expr e = parse_expression ();
  (void)e;
  match (TOK_SEMI);
}

static void
parse_block (void)
{
  expect (TOK_LBR, "Missing {");
  enter_scope ();
  while (g_cur.type != TOK_RBR && g_cur.type != TOK_EOF)
    {
      parse_statement ();
    }
  expect (TOK_RBR, "Missing }");
  leave_scope ();
  match (TOK_SEMI);
}

static ValueType
parse_type_spec (void)
{
  if (g_cur.type == TOK_INT)
    {
      advance_token ();
      return TYPE_INT;
    }
  if (g_cur.type == TOK_FLOAT)
    {
      advance_token ();
      return TYPE_FLOAT;
    }
  if (g_cur.type == TOK_VOID)
    {
      advance_token ();
      return TYPE_VOID;
    }
  record_error ("Expected type");
  return TYPE_UNKNOWN;
}

static void
parse_params (void)
{
  if (g_cur.type == TOK_RPA)
    {
      return;
    }
  while (1)
    {
      ValueType t = parse_type_spec ();
      if (g_cur.type != TOK_ID)
        {
          record_error ("Expected parameter name");
          return;
        }
      char name[MAX_NAME];
      strncpy (name, g_cur.lexeme, MAX_NAME - 1);
      name[MAX_NAME - 1] = '\0';
      advance_token ();
      if (!declare_symbol (name, t, 0, 0))
        {
          char msg[MAX_NAME * 2];
          snprintf (msg, sizeof (msg), "Duplicate declaration of '%s'", name);
          record_error (msg);
        }
      if (match (TOK_COMMA))
        {
          if (g_cur.type == TOK_RPA)
            {
              record_error ("Trailing comma in params");
              return;
            }
          continue;
        }
      break;
    }
}

static void
parse_function (void)
{
  ValueType ret_type = parse_type_spec ();
  if (g_cur.type != TOK_ID)
    {
      record_error ("Expected function name");
      return;
    }
  char name[MAX_NAME];
  strncpy (name, g_cur.lexeme, MAX_NAME - 1);
  name[MAX_NAME - 1] = '\0';
  advance_token ();

  expect (TOK_LPA, "Missing ( in function");
  enter_scope ();
  g_func_ret = ret_type;
  parse_params ();
  expect (TOK_RPA, "Missing ) in function");
  parse_block ();
  leave_scope ();
  (void)name;
}

static void
parse_toplevel (void)
{
  while (g_cur.type != TOK_EOF)
    {
      if (g_cur.type == TOK_INT || g_cur.type == TOK_FLOAT
          || g_cur.type == TOK_VOID)
        {
          parse_function ();
        }
      else
        {
          parse_statement ();
        }
    }
}

static const char *
print_arg (const char *s)
{
  return s[0] ? s : " ";
}

static void
print_quads (void)
{
  for (int i = 0; i < g_quad_count; ++i)
    {
      Quad *q = &g_quads[i];
      printf ("%d. (%s, %s, %s, %s)\n", i + 1, print_arg (q->op),
              print_arg (q->arg1), print_arg (q->arg2), print_arg (q->res));
    }
}

int
main (int argc, char **argv)
{
  const char *path = NULL;
  if (argc >= 2)
    {
      path = argv[1];
    }
  FILE *fp = stdin;
  if (path)
    {
      fp = fopen (path, "r");
      if (!fp)
        {
          fprintf (stderr, "Cannot open input file: %s\n", path);
          return 1;
        }
    }

  char *src = calloc (1, MAX_SRC);
  if (!src)
    {
      fprintf (stderr, "Out of memory\n");
      return 1;
    }
  size_t nread = fread (src, 1, MAX_SRC - 1, fp);
  src[nread] = '\0';
  if (fp != stdin)
    {
      fclose (fp);
    }

  lexer_init (src);
  advance_token ();
  parse_toplevel ();

  if (g_error_count > 0)
    {
      for (int i = 0; i < g_error_count; ++i)
        {
          fprintf (stderr, "Error: %s\n", g_errors[i]);
        }
      free (src);
      return 1;
    }

  print_quads ();
  free (src);
  return 0;
}
