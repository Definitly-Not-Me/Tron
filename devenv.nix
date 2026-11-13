{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  env.GREET = "devenv";

  packages = [
    pkgs.scc
    pkgs.gum
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = pkgs.python3.withPackages (
      pypkg: with pypkg; [
        pygame
        pytest
        gymnasium
      ]
    );
    venv = {
      enable = true;
      requirements = ''
        ppe-debug
      '';
    };
    uv = {
      enable = true;
    };
  };

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.lgtm.exec = "pytest";

  # https://devenv.sh/basics/
  enterShell = "";

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = "";

  git-hooks.hooks = {
    ruff.enable = true;
    ruff-format.enable = true;
    commitizen.enable = true;

  };

  # See full reference at https://devenv.sh/reference/options/
}
