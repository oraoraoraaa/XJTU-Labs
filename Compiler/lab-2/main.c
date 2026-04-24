#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEXEME 256

typedef enum
{
  TK_ID,
  TK_INT,
  TK_FLO,
  TK_ADD,
  TK_SUB,
  TK_MUL,
  TK_DIV,
  TK_ROP,
  TK_ASG,
  TK_LPA,
  TK_RPA,
  TK_LBK,
  TK_RBK,
  TK_LBR,
  TK_RBR,
  TK_CMA,
  TK_SCO,
  TK_IF,
  TK_ELSE,
  TK_WHILE,
  TK_RETURN,
  TK_INPUT,
  TK_PRINT,
  TK_VOID,
  TK_FLOAT_KW,
  TK_UNKNOWN
} TokenType;

typedef struct
{
  TokenType type;
  char lexeme[MAX_LEXEME];
  int line;
  int col;
} Token;

typedef struct
{
  FILE *fp;
  int line;
  int col;
  int ch;
} Scanner;

static const char *
token_name (TokenType t)
{
  switch (t)
    {
    case TK_ID:
      return "ID";
    case TK_INT:
      return "INT";
    case TK_FLO:
      return "FLO";
    case TK_ADD:
      return "ADD";
    case TK_SUB:
      return "SUB";
    case TK_MUL:
      return "MUL";
    case TK_DIV:
      return "DIV";
    case TK_ROP:
      return "ROP";
    case TK_ASG:
      return "ASG";
    case TK_LPA:
      return "LPA";
    case TK_RPA:
      return "RPA";
    case TK_LBK:
      return "LBK";
    case TK_RBK:
      return "RBK";
    case TK_LBR:
      return "LBR";
    case TK_RBR:
      return "RBR";
    case TK_CMA:
      return "CMA";
    case TK_SCO:
      return "SCO";
    case TK_IF:
      return "IF";
    case TK_ELSE:
      return "ELSE";
    case TK_WHILE:
      return "WHILE";
    case TK_RETURN:
      return "RETURN";
    case TK_INPUT:
      return "INPUT";
    case TK_PRINT:
      return "PRINT";
    case TK_VOID:
      return "VOID";
    case TK_FLOAT_KW:
      return "FLOAT";
    default:
      return "UNKNOWN";
    }
}

static int
is_value_token (TokenType t)
{
  return t == TK_ID || t == TK_INT || t == TK_FLO || t == TK_RPA || t == TK_RBK
         || t == TK_RBR;
}

static void
scanner_init (Scanner *s, FILE *fp)
{
  s->fp = fp;
  s->line = 1;
  s->col = 0;
  s->ch = fgetc (fp);
}

static void
scanner_advance (Scanner *s)
{
  if (s->ch == '\n')
    {
      s->line++;
      s->col = 0;
    }
  else
    {
      s->col++;
    }
  s->ch = fgetc (s->fp);
}

static int
peek_char (Scanner *s)
{
  int c = fgetc (s->fp);
  if (c != EOF)
    {
      ungetc (c, s->fp);
    }
  return c;
}

static void
skip_blanks_and_comments (Scanner *s)
{
  while (s->ch != EOF)
    {
      if (isspace ((unsigned char)s->ch))
        {
          scanner_advance (s);
          continue;
        }

      if (s->ch == '/')
        {
          int nxt = peek_char (s);
          if (nxt == '/')
            {
              scanner_advance (s);
              scanner_advance (s);
              while (s->ch != EOF && s->ch != '\n')
                {
                  scanner_advance (s);
                }
              continue;
            }
          if (nxt == '*')
            {
              scanner_advance (s);
              scanner_advance (s);
              while (s->ch != EOF)
                {
                  if (s->ch == '*')
                    {
                      scanner_advance (s);
                      if (s->ch == '/')
                        {
                          scanner_advance (s);
                          break;
                        }
                    }
                  else
                    {
                      scanner_advance (s);
                    }
                }
              continue;
            }
        }

      break;
    }
}

static TokenType
keyword_or_id (const char *s)
{
  if (strcmp (s, "int") == 0)
    {
      return TK_INT;
    }
  if (strcmp (s, "float") == 0)
    {
      return TK_FLOAT_KW;
    }
  if (strcmp (s, "void") == 0)
    {
      return TK_VOID;
    }
  if (strcmp (s, "if") == 0)
    {
      return TK_IF;
    }
  if (strcmp (s, "else") == 0)
    {
      return TK_ELSE;
    }
  if (strcmp (s, "while") == 0)
    {
      return TK_WHILE;
    }
  if (strcmp (s, "return") == 0)
    {
      return TK_RETURN;
    }
  if (strcmp (s, "input") == 0)
    {
      return TK_INPUT;
    }
  if (strcmp (s, "print") == 0)
    {
      return TK_PRINT;
    }
  return TK_ID;
}

static Token
make_simple (Scanner *s, TokenType type, char c)
{
  Token t;
  t.type = type;
  t.line = s->line;
  t.col = s->col + 1;
  t.lexeme[0] = c;
  t.lexeme[1] = '\0';
  scanner_advance (s);
  return t;
}

static Token
scan_identifier (Scanner *s)
{
  Token t;
  int i = 0;
  t.line = s->line;
  t.col = s->col + 1;

  while (s->ch != EOF
         && (isalpha ((unsigned char)s->ch) || isdigit ((unsigned char)s->ch)
             || s->ch == '_'))
    {
      if (i < MAX_LEXEME - 1)
        {
          t.lexeme[i++] = (char)s->ch;
        }
      scanner_advance (s);
    }
  t.lexeme[i] = '\0';
  t.type = keyword_or_id (t.lexeme);
  return t;
}

static Token
scan_number (Scanner *s, int allow_sign)
{
  Token t;
  int i = 0;
  int has_dot = 0;
  int digits_before_dot = 0;
  int digits_after_dot = 0;

  t.line = s->line;
  t.col = s->col + 1;

  if (allow_sign && (s->ch == '+' || s->ch == '-'))
    {
      t.lexeme[i++] = (char)s->ch;
      scanner_advance (s);
    }

  while (s->ch != EOF && isdigit ((unsigned char)s->ch))
    {
      if (i < MAX_LEXEME - 1)
        {
          t.lexeme[i++] = (char)s->ch;
        }
      digits_before_dot++;
      scanner_advance (s);
    }

  if (s->ch == '.')
    {
      has_dot = 1;
      if (i < MAX_LEXEME - 1)
        {
          t.lexeme[i++] = '.';
        }
      scanner_advance (s);

      while (s->ch != EOF && isdigit ((unsigned char)s->ch))
        {
          if (i < MAX_LEXEME - 1)
            {
              t.lexeme[i++] = (char)s->ch;
            }
          digits_after_dot++;
          scanner_advance (s);
        }
    }

  t.lexeme[i] = '\0';

  if (has_dot && (digits_before_dot > 0 || digits_after_dot > 0))
    {
      t.type = TK_FLO;
    }
  else
    {
      t.type = TK_INT;
    }

  return t;
}

static Token
scan_next_token (Scanner *s, TokenType prev)
{
  skip_blanks_and_comments (s);

  if (s->ch == EOF)
    {
      Token end;
      end.type = TK_UNKNOWN;
      end.lexeme[0] = '\0';
      end.line = s->line;
      end.col = s->col + 1;
      return end;
    }

  if (isalpha ((unsigned char)s->ch) || s->ch == '_')
    {
      return scan_identifier (s);
    }

  if (isdigit ((unsigned char)s->ch)
      || ((s->ch == '+' || s->ch == '-') && !is_value_token (prev)
          && (isdigit ((unsigned char)peek_char (s)) || peek_char (s) == '.'))
      || (s->ch == '.' && isdigit ((unsigned char)peek_char (s))))
    {
      int allow_sign = (s->ch == '+' || s->ch == '-');
      return scan_number (s, allow_sign);
    }

  if (s->ch == '<' || s->ch == '>' || s->ch == '=' || s->ch == '!')
    {
      Token t;
      t.line = s->line;
      t.col = s->col + 1;
      t.lexeme[0] = (char)s->ch;
      t.lexeme[1] = '\0';

      if (s->ch == '=')
        {
          scanner_advance (s);
          if (s->ch == '=')
            {
              t.type = TK_ROP;
              t.lexeme[1] = '=';
              t.lexeme[2] = '\0';
              scanner_advance (s);
            }
          else
            {
              t.type = TK_ASG;
            }
          return t;
        }

      scanner_advance (s);
      if (s->ch == '=')
        {
          t.lexeme[1] = '=';
          t.lexeme[2] = '\0';
          scanner_advance (s);
        }
      t.type = TK_ROP;
      return t;
    }

  if (s->ch == '+')
    {
      return make_simple (s, TK_ADD, '+');
    }
  if (s->ch == '-')
    {
      return make_simple (s, TK_SUB, '-');
    }
  if (s->ch == '*')
    {
      return make_simple (s, TK_MUL, '*');
    }
  if (s->ch == '/')
    {
      return make_simple (s, TK_DIV, '/');
    }
  if (s->ch == '(')
    {
      return make_simple (s, TK_LPA, '(');
    }
  if (s->ch == ')')
    {
      return make_simple (s, TK_RPA, ')');
    }
  if (s->ch == '[')
    {
      return make_simple (s, TK_LBK, '[');
    }
  if (s->ch == ']')
    {
      return make_simple (s, TK_RBK, ']');
    }
  if (s->ch == '{')
    {
      return make_simple (s, TK_LBR, '{');
    }
  if (s->ch == '}')
    {
      return make_simple (s, TK_RBR, '}');
    }
  if (s->ch == ',')
    {
      return make_simple (s, TK_CMA, ',');
    }
  if (s->ch == ';')
    {
      return make_simple (s, TK_SCO, ';');
    }

  Token bad;
  bad.type = TK_UNKNOWN;
  bad.line = s->line;
  bad.col = s->col + 1;
  bad.lexeme[0] = (char)s->ch;
  bad.lexeme[1] = '\0';
  scanner_advance (s);
  return bad;
}

int
main (int argc, char **argv)
{
  const char *input_path = "test.c";
  const char *output_path = "tokens.out";
  if (argc >= 2)
    {
      input_path = argv[1];
    }
  if (argc >= 3)
    {
      output_path = argv[2];
    }

  FILE *in = fopen (input_path, "r");
  if (!in)
    {
      fprintf (stderr, "Cannot open input file: %s\n", input_path);
      return 1;
    }

  FILE *out = fopen (output_path, "w");
  if (!out)
    {
      fprintf (stderr, "Cannot open output file: %s\n", output_path);
      fclose (in);
      return 1;
    }

  Scanner s;
  scanner_init (&s, in);

  TokenType prev = TK_UNKNOWN;
  int error_count = 0;

  while (1)
    {
      Token t = scan_next_token (&s, prev);
      if (t.type == TK_UNKNOWN && t.lexeme[0] == '\0')
        {
          break;
        }

      if (t.type == TK_UNKNOWN)
        {
          fprintf (stderr, "Lexical error at %d:%d, unexpected '%s'\n", t.line,
                   t.col, t.lexeme);
          error_count++;
          prev = TK_UNKNOWN;
          continue;
        }

      fprintf (stdout, "(%s, %s)\n", token_name (t.type), t.lexeme);
      fprintf (out, "(%s, %s)\n", token_name (t.type), t.lexeme);
      prev = t.type;
    }

  fprintf (stdout, "Scanner finished with %d lexical error(s).\n",
           error_count);
  fprintf (out, "Scanner finished with %d lexical error(s).\n", error_count);

  fclose (out);
  fclose (in);
  return 0;
}
