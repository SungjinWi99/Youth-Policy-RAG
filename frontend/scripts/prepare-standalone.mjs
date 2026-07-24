import { cp, mkdir } from "node:fs/promises";
import { join } from "node:path";

const targetRoot = join(".next", "standalone");
const targetNext = join(targetRoot, ".next");

await mkdir(targetNext, { recursive: true });
await cp("public", join(targetRoot, "public"), {
  recursive: true,
  force: true,
});
await cp(join(".next", "static"), join(targetNext, "static"), {
  recursive: true,
  force: true,
});
