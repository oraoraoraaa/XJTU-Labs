#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STATES 128
#define MAX_ALPHABET 64
#define MAX_LINE 1024
#define MAX_STR 256

typedef struct
{
  char symbols[MAX_ALPHABET + 1];
  int alphabet_size;

  int state_count;
  int start_state;

  int accept[MAX_STATES + 1];

  int trans[MAX_STATES + 1][MAX_ALPHABET];
  int loaded;
} DFA;

static void
trim_newline (char *s)
{
  size_t n = strlen (s);
  while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r'))
    {
      s[n - 1] = '\0';
      --n;
    }
}

static void
clear_dfa (DFA *dfa)
{
  int i, j;
  memset (dfa, 0, sizeof (*dfa));
  for (i = 0; i <= MAX_STATES; ++i)
    {
      for (j = 0; j < MAX_ALPHABET; ++j)
        {
          dfa->trans[i][j] = -1;
        }
    }
}

static int
is_blank_line (const char *s)
{
  while (*s)
    {
      if (!isspace ((unsigned char)*s))
        {
          return 0;
        }
      ++s;
    }
  return 1;
}

static int
read_non_empty_line (FILE *fp, char *buf, int sz)
{
  while (fgets (buf, sz, fp))
    {
      trim_newline (buf);
      if (!is_blank_line (buf))
        {
          return 1;
        }
    }
  return 0;
}

static int
index_of_symbol (const DFA *dfa, char c)
{
  int i;
  for (i = 0; i < dfa->alphabet_size; ++i)
    {
      if (dfa->symbols[i] == c)
        {
          return i;
        }
    }
  return -1;
}

static int
parse_ints (const char *line, int *out, int max_out)
{
  int count = 0;
  const char *p = line;

  while (*p && count < max_out)
    {
      while (*p && isspace ((unsigned char)*p))
        {
          ++p;
        }
      if (!*p)
        {
          break;
        }

      char *endptr;
      long v = strtol (p, &endptr, 10);
      if (p == endptr)
        {
          return -1;
        }
      out[count++] = (int)v;
      p = endptr;
    }

  while (*p)
    {
      if (!isspace ((unsigned char)*p))
        {
          return -1;
        }
      ++p;
    }

  return count;
}

static int
load_dfa (const char *path, DFA *dfa, char *err, size_t err_sz)
{
  FILE *fp = fopen (path, "r");
  if (!fp)
    {
      snprintf (err, err_sz, "Cannot open file: %s", path);
      return 0;
    }

  clear_dfa (dfa);

  char line[MAX_LINE];

  if (!read_non_empty_line (fp, line, sizeof (line)))
    {
      snprintf (err, err_sz, "Missing alphabet line.");
      fclose (fp);
      return 0;
    }

  int ai = 0;
  for (int i = 0; line[i] != '\0'; ++i)
    {
      if (!isspace ((unsigned char)line[i]))
        {
          if (ai >= MAX_ALPHABET)
            {
              snprintf (err, err_sz, "Alphabet too large.");
              fclose (fp);
              return 0;
            }
          dfa->symbols[ai++] = line[i];
        }
    }
  dfa->symbols[ai] = '\0';
  dfa->alphabet_size = ai;

  if (dfa->alphabet_size <= 0)
    {
      snprintf (err, err_sz, "Alphabet must not be empty.");
      fclose (fp);
      return 0;
    }

  int nums[MAX_STATES + MAX_ALPHABET + 8];

  if (!read_non_empty_line (fp, line, sizeof (line)))
    {
      snprintf (err, err_sz, "Missing state count line.");
      fclose (fp);
      return 0;
    }
  int cnt = parse_ints (line, nums, 4);
  if (cnt != 1 || nums[0] <= 0 || nums[0] > MAX_STATES)
    {
      snprintf (err, err_sz, "Invalid state count.");
      fclose (fp);
      return 0;
    }
  dfa->state_count = nums[0];

  if (!read_non_empty_line (fp, line, sizeof (line)))
    {
      snprintf (err, err_sz, "Missing start state line.");
      fclose (fp);
      return 0;
    }
  cnt = parse_ints (line, nums, 4);
  if (cnt != 1)
    {
      snprintf (err, err_sz, "Start state must contain exactly one integer.");
      fclose (fp);
      return 0;
    }
  dfa->start_state = nums[0];

  if (!read_non_empty_line (fp, line, sizeof (line)))
    {
      snprintf (err, err_sz, "Missing accept states line.");
      fclose (fp);
      return 0;
    }
  cnt = parse_ints (line, nums, MAX_STATES);
  if (cnt <= 0)
    {
      snprintf (err, err_sz, "Accept states line is invalid or empty.");
      fclose (fp);
      return 0;
    }
  for (int i = 0; i < cnt; ++i)
    {
      if (nums[i] >= 1 && nums[i] <= dfa->state_count)
        {
          dfa->accept[nums[i]] = 1;
        }
    }

  for (int s = 1; s <= dfa->state_count; ++s)
    {
      if (!read_non_empty_line (fp, line, sizeof (line)))
        {
          snprintf (err, err_sz, "Transition table incomplete at state %d.",
                    s);
          fclose (fp);
          return 0;
        }
      cnt = parse_ints (line, nums, MAX_ALPHABET + 8);
      if (cnt != dfa->alphabet_size)
        {
          snprintf (err, err_sz, "State %d transition row needs %d integers.",
                    s, dfa->alphabet_size);
          fclose (fp);
          return 0;
        }
      for (int a = 0; a < dfa->alphabet_size; ++a)
        {
          dfa->trans[s][a] = nums[a];
        }
    }

  fclose (fp);
  dfa->loaded = 1;
  snprintf (err, err_sz, "OK");
  return 1;
}

static int
validate_dfa (const DFA *dfa, char *msg, size_t msg_sz)
{
  if (!dfa->loaded)
    {
      snprintf (msg, msg_sz, "DFA not loaded.");
      return 0;
    }

  if (dfa->alphabet_size <= 0)
    {
      snprintf (msg, msg_sz, "Alphabet is empty.");
      return 0;
    }

  for (int i = 0; i < dfa->alphabet_size; ++i)
    {
      for (int j = i + 1; j < dfa->alphabet_size; ++j)
        {
          if (dfa->symbols[i] == dfa->symbols[j])
            {
              snprintf (msg, msg_sz, "Alphabet has duplicate symbol '%c'.",
                        dfa->symbols[i]);
              return 0;
            }
        }
    }

  if (dfa->start_state < 1 || dfa->start_state > dfa->state_count)
    {
      snprintf (msg, msg_sz, "Start state %d is outside [1, %d].",
                dfa->start_state, dfa->state_count);
      return 0;
    }

  int accept_count = 0;
  for (int s = 1; s <= dfa->state_count; ++s)
    {
      if (dfa->accept[s])
        {
          ++accept_count;
        }
    }
  if (accept_count == 0)
    {
      snprintf (msg, msg_sz, "Accept state set is empty.");
      return 0;
    }

  for (int s = 1; s <= dfa->state_count; ++s)
    {
      for (int a = 0; a < dfa->alphabet_size; ++a)
        {
          int to = dfa->trans[s][a];
          if (to < 1 || to > dfa->state_count)
            {
              snprintf (msg, msg_sz,
                        "Invalid transition: state %d on '%c' -> %d (outside "
                        "[1, %d]).",
                        s, dfa->symbols[a], to, dfa->state_count);
              return 0;
            }
        }
    }

  snprintf (msg, msg_sz, "DFA is valid.");
  return 1;
}

static int
simulate (const DFA *dfa, const char *input, int verbose)
{
  int current = dfa->start_state;

  if (verbose)
    {
      printf ("Start at q%d\n", current);
    }

  for (int i = 0; input[i] != '\0'; ++i)
    {
      int idx = index_of_symbol (dfa, input[i]);
      if (idx < 0)
        {
          if (verbose)
            {
              printf ("Input has symbol '%c' not in alphabet.\n", input[i]);
            }
          return 0;
        }

      int next = dfa->trans[current][idx];
      if (verbose)
        {
          printf ("Read '%c': q%d -> q%d\n", input[i], current, next);
        }
      current = next;
    }

  if (verbose)
    {
      printf ("End at q%d (%s)\n", current,
              dfa->accept[current] ? "ACCEPT" : "REJECT");
    }

  return dfa->accept[current];
}

static void
gen_strings_rec (const DFA *dfa, char *buf, int depth, int max_len,
                 int *accepted_count)
{
  if (depth > max_len)
    {
      return;
    }

  buf[depth] = '\0';
  if (simulate (dfa, buf, 0))
    {
      if (depth == 0)
        {
          printf ("<epsilon>\n");
        }
      else
        {
          printf ("%s\n", buf);
        }
      (*accepted_count)++;
    }

  if (depth == max_len)
    {
      return;
    }

  for (int i = 0; i < dfa->alphabet_size; ++i)
    {
      buf[depth] = dfa->symbols[i];
      gen_strings_rec (dfa, buf, depth + 1, max_len, accepted_count);
    }
}

static void
print_dfa (const DFA *dfa)
{
  if (!dfa->loaded)
    {
      printf ("DFA not loaded.\n");
      return;
    }

  printf ("Alphabet: ");
  for (int i = 0; i < dfa->alphabet_size; ++i)
    {
      putchar (dfa->symbols[i]);
      if (i + 1 < dfa->alphabet_size)
        {
          putchar (' ');
        }
    }
  printf ("\n");

  printf ("States: 1..%d\n", dfa->state_count);
  printf ("Start state: q%d\n", dfa->start_state);

  printf ("Accept states: ");
  int first = 1;
  for (int s = 1; s <= dfa->state_count; ++s)
    {
      if (dfa->accept[s])
        {
          if (!first)
            {
              putchar (' ');
            }
          printf ("q%d", s);
          first = 0;
        }
    }
  printf ("\n");

  printf ("Transition table:\n");
  for (int s = 1; s <= dfa->state_count; ++s)
    {
      printf ("q%d: ", s);
      for (int a = 0; a < dfa->alphabet_size; ++a)
        {
          printf ("%c->q%d", dfa->symbols[a], dfa->trans[s][a]);
          if (a + 1 < dfa->alphabet_size)
            {
              printf (", ");
            }
        }
      printf ("\n");
    }
}

static void
random_string_from_alphabet (const DFA *dfa, int len, char *out, int out_sz)
{
  if (len < 0)
    {
      len = 0;
    }
  if (len >= out_sz)
    {
      len = out_sz - 1;
    }

  for (int i = 0; i < len; ++i)
    {
      int idx = rand () % dfa->alphabet_size;
      out[i] = dfa->symbols[idx];
    }
  out[len] = '\0';
}

int
main (void)
{
  DFA dfa;
  clear_dfa (&dfa);
  srand ((unsigned int)time (NULL));

  printf ("==== DFA Simulator (C) ====\n");
  printf ("Default input file: dfa_in1.dfa\n\n");

  char err[256];
  if (load_dfa ("dfa_in1.dfa", &dfa, err, sizeof (err)))
    {
      printf ("Loaded dfa_in1.dfa successfully.\n");
    }
  else
    {
      printf ("Initial load failed: %s\n", err);
    }

  while (1)
    {
      printf ("\nMenu:\n");
      printf ("1. Load DFA from file\n");
      printf ("2. Show DFA\n");
      printf ("3. Validate DFA\n");
      printf ("4. Print all accepted strings with length <= N\n");
      printf ("5. Judge one input string\n");
      printf ("6. Generate random string and judge\n");
      printf ("0. Exit\n");
      printf ("Select: ");

      int op;
      if (scanf ("%d", &op) != 1)
        {
          printf ("Invalid input.\n");
          break;
        }

      if (op == 0)
        {
          printf ("Bye.\n");
          break;
        }

      if (op == 1)
        {
          char path[MAX_STR];
          printf ("Input DFA file path: ");
          scanf ("%255s", path);
          if (load_dfa (path, &dfa, err, sizeof (err)))
            {
              printf ("Loaded: %s\n", path);
            }
          else
            {
              printf ("Load failed: %s\n", err);
            }
        }
      else if (op == 2)
        {
          print_dfa (&dfa);
        }
      else if (op == 3)
        {
          char msg[256];
          if (validate_dfa (&dfa, msg, sizeof (msg)))
            {
              printf ("VALID: %s\n", msg);
            }
          else
            {
              printf ("INVALID: %s\n", msg);
            }
        }
      else if (op == 4)
        {
          char msg[256];
          if (!validate_dfa (&dfa, msg, sizeof (msg)))
            {
              printf ("Cannot enumerate: %s\n", msg);
              continue;
            }

          int N;
          printf ("Input N (max length): ");
          if (scanf ("%d", &N) != 1 || N < 0 || N > 12)
            {
              printf ("N must be an integer in [0, 12].\n");
              continue;
            }

          char buf[MAX_STR];
          int accepted_count = 0;
          printf ("Accepted strings (length <= %d):\n", N);
          gen_strings_rec (&dfa, buf, 0, N, &accepted_count);
          printf ("Total accepted: %d\n", accepted_count);
        }
      else if (op == 5)
        {
          char msg[256];
          if (!validate_dfa (&dfa, msg, sizeof (msg)))
            {
              printf ("Cannot judge: %s\n", msg);
              continue;
            }

          char input[MAX_STR];
          printf ("Input string: ");
          scanf ("%255s", input);

          int ok = simulate (&dfa, input, 1);
          printf ("Result: %s\n", ok ? "ACCEPT" : "REJECT");
        }
      else if (op == 6)
        {
          char msg[256];
          if (!validate_dfa (&dfa, msg, sizeof (msg)))
            {
              printf ("Cannot judge: %s\n", msg);
              continue;
            }

          int len;
          printf ("Random string length: ");
          if (scanf ("%d", &len) != 1 || len < 0 || len > 200)
            {
              printf ("Length must be in [0, 200].\n");
              continue;
            }

          char input[MAX_STR];
          random_string_from_alphabet (&dfa, len, input, MAX_STR);
          printf ("Generated string: %s\n", input[0] ? input : "<epsilon>");
          int ok = simulate (&dfa, input, 1);
          printf ("Result: %s\n", ok ? "ACCEPT" : "REJECT");
        }
      else
        {
          printf ("Unknown option.\n");
        }
    }

  return 0;
}
